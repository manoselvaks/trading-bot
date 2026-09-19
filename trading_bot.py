"""
Local trading bot script — run this from Terminal with:
    python3 trading_bot.py

Reads Alpaca API keys from a .env file in the same folder (never hardcode
keys directly in a script you might share or back up somewhere).
"""

import os
import csv
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ----------------------------------------------------------------------
# CSV logging setup
# ----------------------------------------------------------------------
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_log.csv")
LOG_COLUMNS = [
    "timestamp", "ticker", "signal", "action_taken",
    "price", "notional_or_qty", "account_equity", "daily_pnl_pct", "note"
]

def log_row(ticker, signal, action_taken, price, notional_or_qty, account_equity, daily_pnl_pct, note=""):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            ticker, signal, action_taken,
            round(float(price), 2) if price != "" else "",
            round(float(notional_or_qty), 2) if notional_or_qty != "" else "",
            round(float(account_equity), 2),
            f"{daily_pnl_pct:.4f}", note
        ])

# ----------------------------------------------------------------------
# Load API keys from .env
# ----------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing API keys — check that .env exists and has both keys set.")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
TICKER = "AAPL"
SHORT_WINDOW = 20
LONG_WINDOW = 50

RISK_PER_TRADE_PCT = 0.01
STOP_LOSS_PCT = 0.05
MAX_DAILY_LOSS_PCT = 0.03

# ----------------------------------------------------------------------
# 1. Circuit breaker
# ----------------------------------------------------------------------
account = trading_client.get_account()
equity_now = float(account.equity)
equity_yesterday_close = float(account.last_equity)
daily_pnl_pct = (equity_now - equity_yesterday_close) / equity_yesterday_close

print(f"Today's P&L so far: {daily_pnl_pct:.2%}")

if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:
    print(f"CIRCUIT BREAKER TRIPPED: down {daily_pnl_pct:.2%} today "
          f"(limit is {MAX_DAILY_LOSS_PCT:.0%}). No new trades today.")
    STOP_ALL_TRADING = True
else:
    STOP_ALL_TRADING = False
    print("Circuit breaker OK - clear to continue.")

# ----------------------------------------------------------------------
# 2. Check position + pending orders
# ----------------------------------------------------------------------
positions = trading_client.get_all_positions()
current_position = next((p for p in positions if p.symbol == TICKER), None)

open_orders = trading_client.get_orders()
pending_order_exists = any(
    o.symbol == TICKER and str(o.status) in ("OrderStatus.NEW", "OrderStatus.PENDING_NEW", "OrderStatus.ACCEPTED")
    for o in open_orders
)

if pending_order_exists:
    print(f"There is already a pending order for {TICKER} - skipping new entry to avoid stacking.")

if current_position is not None:
    # Stop-loss runs even if the circuit breaker has tripped — the circuit
    # breaker only blocks *new* entries; it should never suppress closing
    # a losing position, since a bad day is exactly when that matters most.
    entry_price = float(current_position.avg_entry_price)
    current_price = float(current_position.current_price)
    unrealized_pct = (current_price - entry_price) / entry_price

    print(f"{TICKER} position: entry ${entry_price:.2f}, "
          f"current ${current_price:.2f} ({unrealized_pct:.2%})")

    if unrealized_pct <= -STOP_LOSS_PCT:
        print(f"STOP-LOSS TRIGGERED: {unrealized_pct:.2%} <= -{STOP_LOSS_PCT:.0%}")
        order = MarketOrderRequest(
            symbol=TICKER,
            qty=current_position.qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(order)
        print(f"Placed emergency SELL order to close {current_position.qty} shares")
        log_row(TICKER, "STOP_LOSS", "SELL", current_price, current_position.qty,
                equity_now, daily_pnl_pct, note=f"stop-loss hit at {unrealized_pct:.2%}")
        current_position = None

# ----------------------------------------------------------------------
# 3. Compute strategy signal
# ----------------------------------------------------------------------
data = yf.download(TICKER, period="6mo", progress=False)
close = data["Close"].squeeze()
ma_short = close.rolling(window=SHORT_WINDOW).mean()
ma_long = close.rolling(window=LONG_WINDOW).mean()
current_signal = "BUY" if ma_short.iloc[-1] > ma_long.iloc[-1] else "SELL"
print(f"Signal: {current_signal}")

# ----------------------------------------------------------------------
# 4. Position-sized entry
# ----------------------------------------------------------------------
holding = current_position is not None

if STOP_ALL_TRADING:
    print("No trades placed - circuit breaker is active.")
    log_row(TICKER, current_signal, "BLOCKED", close.iloc[-1], "",
            equity_now, daily_pnl_pct, note="circuit breaker tripped")

elif pending_order_exists:
    log_row(TICKER, current_signal, "SKIPPED", close.iloc[-1], "",
            equity_now, daily_pnl_pct, note="pending order already exists")

elif current_signal == "BUY" and not holding:
    dollars_at_risk = equity_now * RISK_PER_TRADE_PCT
    position_dollars = dollars_at_risk / STOP_LOSS_PCT
    position_dollars = min(position_dollars, equity_now * 0.20)

    order = MarketOrderRequest(
        symbol=TICKER,
        notional=round(position_dollars, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    trading_client.submit_order(order)
    print(f"Placed BUY order for ${position_dollars:.2f} of {TICKER} "
          f"(sized to risk {RISK_PER_TRADE_PCT:.0%} of account "
          f"given a {STOP_LOSS_PCT:.0%} stop-loss)")
    log_row(TICKER, current_signal, "BUY", close.iloc[-1], round(position_dollars, 2),
            equity_now, daily_pnl_pct)

elif current_signal == "SELL" and holding:
    order = MarketOrderRequest(
        symbol=TICKER,
        qty=current_position.qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    trading_client.submit_order(order)
    print(f"Placed SELL order to close {current_position.qty} shares")
    log_row(TICKER, current_signal, "SELL", close.iloc[-1], current_position.qty,
            equity_now, daily_pnl_pct)

else:
    print("No action needed - position already matches signal.")
    log_row(TICKER, current_signal, "NONE", close.iloc[-1], "",
            equity_now, daily_pnl_pct, note="already correctly positioned")
