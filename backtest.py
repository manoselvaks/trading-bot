"""
Backtests the moving-average crossover strategy used by trading_bot.py
against simple buy-and-hold over the same historical period, with
transaction costs applied on every trade.

This is the backtest referenced in the README — it's what justified
sticking with the simple crossover strategy in trading_bot.py instead of
the ML classifier explored in experiments/overfitting_experiment.py.

Usage:
    python3 backtest.py AAPL
    python3 backtest.py AAPL --period 3y --short 20 --long 50 --cost-bps 5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252


def compute_signal(close, short_window, long_window):
    """1 = long (short MA above long MA), 0 = flat. Uses .shift(1) so the
    signal for day t is decided using only data available through day
    t-1 — trading on today's close using today's own MA would be
    look-ahead bias, since you can't know today's closing average until
    the close has already happened."""
    ma_short = close.rolling(short_window).mean()
    ma_long = close.rolling(long_window).mean()
    signal = (ma_short > ma_long).astype(int)
    return signal.shift(1).fillna(0)


def run_backtest(close, short_window, long_window, cost_bps):
    """Returns a DataFrame with daily returns for the strategy and for
    buy-and-hold, both net of the strategy's transaction costs. cost_bps
    is charged only on days the position actually changes (a trade),
    not on every day the strategy happens to be in the market."""
    daily_return = close.pct_change().fillna(0)
    position = compute_signal(close, short_window, long_window)

    trade_occurred = position.diff().fillna(position).abs()  # 1 on entry/exit days
    cost = trade_occurred * (cost_bps / 10_000)

    strategy_return = position * daily_return - cost
    buy_hold_return = daily_return

    return pd.DataFrame({
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "position": position,
    })


def summarize(returns, label):
    """Standard backtest stats: total/annualized return, Sharpe, max
    drawdown — the same numbers you'd report for any strategy, not just
    this one."""
    equity_curve = (1 + returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1

    n_days = len(returns)
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (equity_curve.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0.0

    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (returns.mean() * TRADING_DAYS_PER_YEAR) / ann_vol if ann_vol > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    print(f"\n{label}")
    print(f"  Total return:     {total_return:.2%}")
    print(f"  CAGR:             {cagr:.2%}")
    print(f"  Annualized vol:   {ann_vol:.2%}")
    print(f"  Sharpe ratio:     {sharpe:.2f}")
    print(f"  Max drawdown:     {max_drawdown:.2%}")

    return {
        "total_return": total_return, "cagr": cagr,
        "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_drawdown,
    }


def plot_results(close, results, short_window, long_window, ticker, output_path):
    """Two-panel chart: price with both moving averages on top, growth of
    $1 under each approach on the bottom — the standard way to show a
    crossover backtest visually rather than as numbers alone."""
    ma_short = close.rolling(short_window).mean()
    ma_long = close.rolling(long_window).mean()
    strategy_equity = (1 + results["strategy_return"]).cumprod()
    buyhold_equity = (1 + results["buy_hold_return"]).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(close.index, close, label="Price", color="black", linewidth=1)
    axes[0].plot(close.index, ma_short, label=f"{short_window}-day MA", color="tab:blue")
    axes[0].plot(close.index, ma_long, label=f"{long_window}-day MA", color="tab:orange")
    axes[0].set_title(f"{ticker} price with moving-average crossover")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend(loc="upper left")

    axes[1].plot(buyhold_equity.index, buyhold_equity, label="Buy & hold", color="gray")
    axes[1].plot(strategy_equity.index, strategy_equity, label="MA crossover strategy", color="tab:blue")
    axes[1].set_title("Growth of $1 invested")
    axes[1].set_ylabel("Equity ($)")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nChart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest the MA-crossover strategy vs. buy-and-hold."
    )
    parser.add_argument("ticker")
    parser.add_argument("--period", default="3y",
                         help="History window to test over (default: 3y)")
    parser.add_argument("--short", type=int, default=20)
    parser.add_argument("--long", type=int, default=50)
    parser.add_argument("--cost-bps", type=float, default=5.0,
                         help="Round-trip transaction cost in basis points "
                              "charged per trade (default: 5bps)")
    parser.add_argument("--plot", default=None,
                         help="Save a price/equity-curve chart to this path "
                              "(default: <TICKER>_backtest.png)")
    parser.add_argument("--no-plot", action="store_true",
                         help="Skip saving a chart, print stats only")
    args = parser.parse_args()

    data = yf.download(args.ticker, period=args.period, progress=False)
    close = data["Close"].squeeze()

    results = run_backtest(close, args.short, args.long, args.cost_bps)

    print(f"Backtest: {args.ticker} | {args.period} history | "
          f"{args.short}/{args.long}-day crossover | "
          f"{args.cost_bps:.0f}bps cost per trade")

    strategy_stats = summarize(results["strategy_return"], "Moving-average crossover")
    buyhold_stats = summarize(results["buy_hold_return"], "Buy-and-hold")

    n_trades = int(results["position"].diff().fillna(0).abs().sum())
    print(f"\nNumber of trades: {n_trades}")

    if strategy_stats["cagr"] > buyhold_stats["cagr"]:
        print("\nConclusion: the crossover strategy beat buy-and-hold over this "
              "period, net of costs.")
    else:
        print("\nConclusion: buy-and-hold beat the crossover strategy over this "
              "period, net of costs — a common result once trading costs are "
              "included, and worth reporting honestly rather than only "
              "backtesting periods that happen to favor the strategy.")

    if not args.no_plot:
        output_path = args.plot or f"{args.ticker.upper()}_backtest.png"
        plot_results(close, results, args.short, args.long, args.ticker.upper(), output_path)


if __name__ == "__main__":
    main()
