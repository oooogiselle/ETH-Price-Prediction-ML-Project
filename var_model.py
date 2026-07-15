"""var_model.py -- Vector Autoregression across close price and volume."""

from statsmodels.tsa.api import VAR
from preprocessing import load_clean_data, train_test_split_series
from utils import report, plot_predictions

COLUMNS = ["close", "volume"]


def run():
    df = load_clean_data()[COLUMNS]
    train, test = train_test_split_series(df.iloc[:, 0])  # just for index/length
    train_df, test_df = df.iloc[:len(train)], df.iloc[len(train):]

    model = VAR(train_df).fit(maxlags=15, ic="aic")
    forecast = model.forecast(train_df.values[-model.k_ar:], steps=len(test_df))

    preds_close = forecast[:, 0]

    result = report("VAR", test_df["close"], preds_close)
    plot_predictions(test_df["close"], preds_close, "VAR Model - ETH-USD", "imgs/var_results.png")
    return result


if __name__ == "__main__":
    run()
