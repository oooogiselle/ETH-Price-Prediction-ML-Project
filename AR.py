"""AR.py -- fits a pure autoregressive model (ARIMA order (p,0,0))."""

from statsmodels.tsa.arima.model import ARIMA
from preprocessing import load_clean_data, train_test_split_series, TARGET_COL
from utils import report, plot_predictions

ORDER = (5, 0, 0)  # p from auto_arima.py; d=0, q=0 -> pure AR


def run():
    df = load_clean_data()
    train, test = train_test_split_series(df[TARGET_COL])

    model = ARIMA(train, order=ORDER).fit()
    preds = model.forecast(steps=len(test))
    preds.index = test.index

    report("AR", test, preds)
    plot_predictions(test, preds, "AR Model - ETH-USD", "imgs/ar_results.png")


if __name__ == "__main__":
    run()
