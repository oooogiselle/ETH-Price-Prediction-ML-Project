"""
visualize.py -- generates visuals that are specific to what THIS project
actually does differently, rather than generic time-series plots:

  1. candlestick_volume.png   -- OHLCV candlestick chart (the raw data is
                                   OHLCV, not just closing price -- show it)
  2. garch_volatility.png     -- GARCH(1,1) conditional volatility over time,
                                   the specific reason the GARCH-SARIMAX
                                   model exists (crypto volatility clustering)
  3. model_family_comparison.png -- avg RMSE grouped by model FAMILY
                                   (classical stats / regression / tree
                                   ensembles / deep learning) -- this is the
                                   actual point of the BTC->ETH variation:
                                   do the new model families beat the old ones?
  4. feature_importance.png   -- which lagged days matter most to XGBoost
                                   when predicting next-day close
  5. error_vs_volatility.png  -- XGBoost's prediction error over time,
                                   overlaid against realized volatility --
                                   does the model get worse specifically
                                   during volatile stretches?

Run after data_loader.py. Optionally pass run_all.py's collected results
into plot_model_family_comparison(). Output goes to imgs/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from xgboost import XGBRegressor

from preprocessing import load_clean_data, make_supervised_features, train_test_split_series, TARGET_COL, difference

os.makedirs("imgs", exist_ok=True)

FAMILY_MAP = {
    "AR": "Classical Stats", "ARMA": "Classical Stats", "ARIMA": "Classical Stats",
    "SARIMAX": "Classical Stats", "GARCH-SARIMAX": "Classical Stats", "VAR": "Classical Stats",
    "ElasticNet": "Linear Regression", "BayesianRidge": "Linear Regression", "PolynomialRegression": "Linear Regression",
    "RandomForest": "Tree Ensembles", "XGBoost": "Tree Ensembles",
    "LSTM": "Deep Learning",
}


def plot_candlestick_volume(df, n_days=90):
    """OHLC candlestick + volume bars for the most recent n_days."""
    recent = df.tail(n_days).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    width = 0.6
    for date, row in recent.iterrows():
        color = "#2ca02c" if row["close"] >= row["open"] else "#d62728"
        ax1.plot([date, date], [row["low"], row["high"]], color=color, linewidth=0.8)
        ax1.add_patch(plt.Rectangle(
            (mdates.date2num(date) - width / 2, min(row["open"], row["close"])),
            width, abs(row["close"] - row["open"]) or 0.01,
            color=color,
        ))
    ax1.set_title(f"ETH-USD OHLC Candlestick (last {n_days} days)")
    ax1.set_ylabel("Price (USD)")
    ax1.xaxis_date()

    ax2.bar(recent.index, recent["volume"], color="#7f7f7f", width=width)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("imgs/candlestick_volume.png")
    plt.close()
    print("Saved imgs/candlestick_volume.png")


def plot_garch_volatility(df):
    """Fits GARCH(1,1) on daily returns and plots conditional volatility --
    the direct visual justification for why GARCH-SARIMAX is in the model suite."""
    returns = 100 * df[TARGET_COL].pct_change().dropna()
    garch = arch_model(returns, vol="Garch", p=1, q=1).fit(disp="off")
    cond_vol = garch.conditional_volatility

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(df.index[1:], returns, color="#1f77b4", linewidth=0.6)
    ax1.set_title("ETH-USD Daily Returns (%)")
    ax1.set_ylabel("Return %")

    ax2.plot(cond_vol.index, cond_vol, color="#d62728")
    ax2.set_title("GARCH(1,1) Conditional Volatility -- volatility clustering visible here\nis exactly what SARIMAX alone can't capture")
    ax2.set_ylabel("Conditional Volatility")
    ax2.set_xlabel("Date")
    plt.tight_layout()
    plt.savefig("imgs/garch_volatility.png")
    plt.close()
    print("Saved imgs/garch_volatility.png")


def plot_model_family_comparison(results):
    """
    results: list of dicts like {"model": "XGBoost", "rmse": 123.4, ...}
    (exactly what utils.report() returns / run_all.py collects)

    Groups models into families (Classical Stats / Linear Regression /
    Tree Ensembles / Deep Learning) and plots average RMSE per family --
    this answers the actual question the BTC->ETH variation raises: do
    the newly-added model families (tree ensembles, deep learning)
    outperform the classical time-series suite from the original project?
    """
    df = pd.DataFrame(results)
    df["family"] = df["model"].map(FAMILY_MAP).fillna("Other")
    family_avg = df.groupby("family")["rmse"].mean().sort_values()

    colors = {"Classical Stats": "#7f7f7f", "Linear Regression": "#1f77b4",
              "Tree Ensembles": "#2ca02c", "Deep Learning": "#d62728", "Other": "#9467bd"}
    bar_colors = [colors.get(f, "#9467bd") for f in family_avg.index]

    plt.figure(figsize=(9, 5))
    plt.bar(family_avg.index, family_avg.values, color=bar_colors)
    plt.title("Average RMSE by Model Family -- classical vs. newly-added methods")
    plt.ylabel("Avg RMSE (USD)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("imgs/model_family_comparison.png")
    plt.close()
    print("Saved imgs/model_family_comparison.png")


def plot_feature_importance(df):
    """Which lagged day matters most to XGBoost when predicting tomorrow's close."""
    feats = make_supervised_features(df, lags=10)
    train, _ = train_test_split_series(feats)
    X_train, y_train = train.drop(columns="target"), train["target"]

    model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()

    plt.figure(figsize=(9, 5))
    plt.barh(importances.index, importances.values, color="#2ca02c")
    plt.title("XGBoost Feature Importance -- which past day matters most?")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("imgs/feature_importance.png")
    plt.close()
    print("Saved imgs/feature_importance.png")
    return model


def plot_error_vs_volatility(df, model=None):
    """Plots XGBoost's absolute prediction error over time against realized
    (rolling 7-day) volatility -- shows whether the model specifically
    struggles during volatile stretches, which matters a lot for a coin
    as volatile as ETH."""
    feats = make_supervised_features(df, lags=10)
    train, test = train_test_split_series(feats)
    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    if model is None:
        model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    abs_error = np.abs(y_test.values - preds)

    realized_vol = df[TARGET_COL].pct_change().rolling(7).std().reindex(test.index) * 100

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(test.index, abs_error, color="#d62728", label="XGBoost abs. error (USD)")
    ax1.set_ylabel("Absolute Error (USD)", color="#d62728")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(test.index, realized_vol, color="#1f77b4", alpha=0.6, label="7-day realized volatility (%)")
    ax2.set_ylabel("Realized Volatility (%)", color="#1f77b4")

    plt.title("Does XGBoost's error track ETH's volatility?")
    fig.tight_layout()
    plt.savefig("imgs/error_vs_volatility.png")
    plt.close()
    print("Saved imgs/error_vs_volatility.png")


def main():
    df = load_clean_data()
    plot_candlestick_volume(df)
    plot_garch_volatility(df)
    model = plot_feature_importance(df)
    plot_error_vs_volatility(df, model=model)
    print("\nRun run_all.py to collect model results, then call "
          "plot_model_family_comparison(results) to get the family comparison chart.")


if __name__ == "__main__":
    main()