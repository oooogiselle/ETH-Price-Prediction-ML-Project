"""
GARCH_SARIMAX.py -- fits SARIMAX for the conditional mean, then fits a GARCH(1,1)
model on the SARIMAX residuals to capture volatility clustering, combining
both into a final forecast. This was the best-performing model in the
original BTC project and we keep it as the volatility-aware baseline here.
"""

from arch import arch_model
from SARIMAX import run as run_sarimax
from utils import report, plot_predictions


def run():
    sarimax_result = run_sarimax()
    model = sarimax_result["fitted_model"]
    test_y = sarimax_result["test_y"]
    sarimax_preds = sarimax_result["preds"]
    resid = model.resid.dropna()

    garch = arch_model(resid, vol="Garch", p=1, q=1).fit(disp="off")
    garch_forecast = garch.forecast(horizon=len(test_y))
    vol_adjustment = garch_forecast.mean.values[-1]

    combined_preds = sarimax_preds + vol_adjustment
    combined_preds.index = test_y.index

    result = report("GARCH-SARIMAX", test_y, combined_preds)
    plot_predictions(test_y, combined_preds, "GARCH+SARIMAX Model - ETH-USD", "imgs/garch_sarimax_results.png")
    return result


if __name__ == "__main__":
    run()
