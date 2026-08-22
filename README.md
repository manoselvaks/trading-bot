# Trading Bot

An automated algorithmic trading system built with Python, using a moving-average crossover strategy on paper-traded stocks via the Alpaca API.

## What it does

- Pulls historical price data and computes a 20-day / 50-day moving average crossover signal
- Automatically places or skips trades based on that signal, with duplicate-order protection (checks pending orders before placing new ones)
- Runs unattended every weekday via `cron`, scheduled around market open
- Logs every decision — trade or skip — to `trade_log.csv` for a full history

## Risk management

- **Circuit breaker**: refuses to place new trades if the account is down more than a set % on the day
- **Stop-loss**: automatically closes a position if it drops more than 5% from entry
- **Position sizing**: sizes each trade based on a fixed % of account equity at risk, capped at a maximum position size

## Tech stack

- Python
- [Alpaca API](https://alpaca.markets) (paper trading)
- pandas / numpy for backtesting and data handling
- macOS `cron` for scheduling
- scikit-learn (used in an early ML experiment — see below)

## Backtesting

Includes a proper backtest comparing the moving-average strategy against simple buy-and-hold over the same period, using realistic transaction cost assumptions.

There's also a deliberate ML experiment in the project's history: a Random Forest classifier trained on price-derived features to predict next-day direction. It hit 63% accuracy on training data but only ~50% (coin-flip) on unseen test data — a concrete, hands-on demonstration of overfitting, and part of why the bot sticks to the simpler, more transparent moving-average approach in production.

## Setup

```bash
git clone <repo-url>
cd trading-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (not committed — see `.gitignore`) with your Alpaca API credentials:

```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

Run it manually:

```bash
python3 trading_bot.py
```

Or schedule it via `cron` to run automatically on weekdays.

## Status

Currently running in **paper trading mode only** — no real money is involved. This is a learning project exploring algorithmic trading, risk management, and the practical pitfalls of applying ML to financial data.

## Disclaimer

This project is for educational purposes. Nothing here is financial advice, and past backtest performance is not indicative of future results.
