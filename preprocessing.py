"""
preprocessing.py

Shared preprocessing utilities used by every model script:
- load the cached CSV
- basic cleaning
- stationarity check (ADF test) + order-1 differencing
- train/test split (chronological, no shuffling -- this is a time series)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

DATA_PATH = "data/eth_ohlcv.csv"
TARGET_COL = "close"
TEST_SIZE = 0.15  # last 15% of the series held out for testing


def load_clean_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]
    return df


def adf_test(series, label=""):
    result = adfuller(series.dropna())
    print(f"ADF test {label}: statistic={result[0]:.4f}, p-value={result[1]:.4f}")
    is_stationary = result[1] < 0.05
    print(f"  -> {'stationary' if is_stationary else 'NOT stationary'} at 5% significance")
    return is_stationary


def difference(series, order=1):
    diff = series.copy()
    for _ in range(order):
        diff = diff.diff().dropna()
    return diff


def train_test_split_series(series, test_size=TEST_SIZE):
    n_test = int(len(series) * test_size)
    train, test = series.iloc[:-n_test], series.iloc[-n_test:]
    return train, test


def make_supervised_features(df, lags=10):
    """
    Turns the raw OHLCV series into a lag-feature table for the
    regression / tree / boosting models (elasticnet, bayesian, polyreg,
    random_forest, xgboost). Each row's features are the previous `lags`
    closing prices; the target is the next day's close.
    """
    data = pd.DataFrame(index=df.index)
    for lag in range(1, lags + 1):
        data[f"lag_{lag}"] = df[TARGET_COL].shift(lag)
    data["target"] = df[TARGET_COL]
    data = data.dropna()
    return data


if __name__ == "__main__":
    df = load_clean_data()
    print(df.describe())
    adf_test(df[TARGET_COL], label="close (levels)")
    adf_test(difference(df[TARGET_COL], 1), label="close (order-1 diff)")
