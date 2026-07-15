"""SARIMAX.py -- fits a Seasonal ARIMA with exogenous regressors (volume)."""

from statsmodels.tsa.statespace.sarimax import SARIMAX
from preprocessing import load_clean_data, train_test_split_series, TARGET_COL
from utils import report, plot_predictions

ORDER = (5, 1, 2)
SEASONAL_ORDER = (1, 0, 1, 7)  # weekly seasonality


def run():
    df = load_clean_data()
    train_y, test_y = train_test_split_series(df[TARGET_COL])
    train_x, test_x = train_test_split_series(df[["volume"]])

    model = SARIMAX(
        train_y, exog=train_x, order=ORDER, seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=True, enforce_invertibility=True,
    ).fit(disp=False)

    preds = model.forecast(steps=len(test_y), exog=test_x)
    preds.index = test_y.index

    result = report("SARIMAX", test_y, preds)
    plot_predictions(test_y, preds, "SARIMAX Model - ETH-USD", "imgs/sarimax_results.png")
    result.update({"fitted_model": model, "test_y": test_y, "preds": preds})
    return result


if __name__ == "__main__":
    run()
