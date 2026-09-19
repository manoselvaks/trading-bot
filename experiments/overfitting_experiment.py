"""
The ML experiment referenced in the README: trains a Random Forest
classifier on price-derived features to predict next-day direction
(up/down), and deliberately reports both training and held-out test
accuracy side by side — because a model that only reports one of those
numbers is hiding the number that actually matters.

This is why trading_bot.py uses a simple moving-average crossover in
production instead: the model below fits the training data far better
than it predicts unseen data, which is the textbook definition of
overfitting on financial time series, where the signal-to-noise ratio is
low and yesterday's pattern often doesn't repeat tomorrow.

Usage:
    python3 experiments/overfitting_experiment.py AAPL
"""

import argparse

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def build_features(close):
    """Price-derived features only — no lookahead, each feature at row t
    uses only information available at the close of day t."""
    df = pd.DataFrame(index=close.index)
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["ma_5"] = close.rolling(5).mean() / close - 1
    df["ma_20"] = close.rolling(20).mean() / close - 1
    df["volatility_10d"] = close.pct_change().rolling(10).std()

    # Target: did price go up the *next* trading day? This is what makes
    # it a next-day direction predictor rather than a same-day one.
    df["target"] = (close.shift(-1) > close).astype(int)

    return df.dropna()


def chronological_train_test_split(df, test_fraction=0.25):
    """A random shuffle split would leak future information into training
    (the model would effectively see the future before predicting the
    past) — time series data must be split chronologically instead."""
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def run_experiment(close):
    df = build_features(close)
    feature_cols = ["return_1d", "return_5d", "return_10d", "ma_5", "ma_20", "volatility_10d"]

    train, test = chronological_train_test_split(df)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(train[feature_cols], train["target"])

    train_accuracy = accuracy_score(train["target"], model.predict(train[feature_cols]))
    test_accuracy = accuracy_score(test["target"], model.predict(test[feature_cols]))

    # A model that always predicted "up" would score this often, purely
    # from the market's historical upward drift — the real bar a
    # direction predictor needs to clear isn't 50%, it's this number.
    naive_baseline = max(test["target"].mean(), 1 - test["target"].mean())

    return train_accuracy, test_accuracy, naive_baseline


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrates overfitting: RF classifier train vs. test accuracy."
    )
    parser.add_argument("ticker")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()

    data = yf.download(args.ticker, period=args.period, progress=False)
    close = data["Close"].squeeze()

    train_acc, test_acc, baseline = run_experiment(close)

    print(f"Next-day direction predictor: {args.ticker}, {args.period} history")
    print(f"  Training accuracy:        {train_acc:.1%}")
    print(f"  Held-out test accuracy:   {test_acc:.1%}")
    print(f"  Naive baseline (always predict the majority class): {baseline:.1%}")
    print()
    if test_acc <= baseline + 0.02:
        print("Conclusion: the gap between training and test accuracy shows the "
              "model memorized noise in the training window rather than learning "
              "a real predictive pattern — test accuracy is barely better than "
              "just always guessing the more common direction. This is why the "
              "bot doesn't use this model in production.")
    else:
        print("Conclusion: test accuracy beat the naive baseline on this run — "
              "worth re-running across multiple tickers and periods before "
              "trusting it, since a single period beating the baseline is not "
              "strong evidence of a real edge.")


if __name__ == "__main__":
    main()
