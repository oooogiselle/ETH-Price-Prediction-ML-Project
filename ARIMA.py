"""ARIMA.py -- fits the full ARIMA(p,d,q) model using the order found by auto_arima.py."""

from statsmodels.tsa.arima.model import ARIMA
from preprocessing import load_clean_data, train_test_split_series, TARGET_COL
from utils import report, plot_predictions

ORDER = (5, 1, 2)  # (p,d,q) from auto_arima.py


def run():
    df = load_clean_data()
    train, test = train_test_split_series(df[TARGET_COL])

    model = ARIMA(train, order=ORDER).fit()
    preds = model.forecast(steps=len(test))
    preds.index = test.index

    result = report("ARIMA", test, preds)
    plot_predictions(test, preds, "ARIMA Model - ETH-USD", "imgs/arima_results.png")
    return result


if __name__ == "__main__":
    run()
