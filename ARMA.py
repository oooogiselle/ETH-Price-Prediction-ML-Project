"""ARMA.py -- fits an ARMA model (ARIMA order (p,0,q))."""

from statsmodels.tsa.arima.model import ARIMA
from preprocessing import load_clean_data, train_test_split_series, TARGET_COL
from utils import report, plot_predictions

ORDER = (5, 0, 2)  # from auto_arima.py, forced d=0 for pure ARMA


def run():
    df = load_clean_data()
    train, test = train_test_split_series(df[TARGET_COL])

    model = ARIMA(train, order=ORDER).fit()
    preds = model.forecast(steps=len(test))
    preds.index = test.index

    result = report("ARMA", test, preds)
    plot_predictions(test, preds, "ARMA Model - ETH-USD", "imgs/arma_results.png")
    return result


if __name__ == "__main__":
    run()
