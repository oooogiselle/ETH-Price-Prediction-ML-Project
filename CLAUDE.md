# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A time-series/ML project predicting ETH-USD prices. It's a variation on an
original BTC price prediction project: data source was swapped from
CoinMarketCap to yfinance, and the model suite was extended with tree
ensembles (Random Forest, XGBoost) and an LSTM on top of the original
classical time-series/regression models.

## Setup and common commands

```bash
pip install -r requirements.txt   # full deps (includes tensorflow)
pip install -r requirements-core.txt   # everything except tensorflow, for a lighter install
```

```bash
python data_loader.py     # fetch ETH-USD OHLCV from yfinance -> data/eth_ohlcv.csv (required before anything else)
python auto_arima.py      # pmdarima grid search for best (p,d,q); update ORDER constants in AR/ARMA/ARIMA/SARIMAX by hand
python run_all.py          # run every model in MODEL_MODULES, print RMSE/MAE/MAPE leaderboard, save imgs/model_family_comparison.png
python visualize.py       # generate candlestick, GARCH volatility, feature importance, error-vs-volatility plots
python <model>.py          # run any single model directly, e.g. python xgboost_model.py
```

There is no test suite, linter, or CI config in this repo.

## Architecture

**Pipeline shape:** `data_loader.py` caches raw OHLCV to `data/eth_ohlcv.csv`
-> `preprocessing.py` loads/cleans it and exposes shared transforms -> each
model script consumes those transforms independently -> `utils.py` provides
shared metrics/plotting used by every model script -> `run_all.py` and
`visualize.py` aggregate across models.

**`preprocessing.py` is the shared contract every model script depends on:**
- `load_clean_data()` — reads the cached CSV, sorts, drops NaNs/dupes
- `train_test_split_series()` — chronological split (last 15% held out), never shuffled, since this is a time series
- `make_supervised_features(df, lags=10)` — builds a lag-feature table (lag_1..lag_N + target) for the regression/tree/boosting models (elasticnet, bayesian, polyreg, random_forest, xgboost_model)
- Classical time-series models (AR, ARMA, ARIMA, SARIMAX, var_model) instead consume the raw `close` series directly and fit statsmodels' own AR/ARIMA machinery

**Every model script follows the same convention:** a module-level `run()`
function that loads data, fits, predicts, calls `report()` and
`plot_predictions()`, and is also invoked from `if __name__ == "__main__"` so
the script works both standalone and when imported by `run_all.py`.

**`GARCH_SARIMAX.py` composes rather than duplicates SARIMAX:** it imports
`SARIMAX.run` and reuses its fitted model/residuals, then layers a GARCH(1,1)
volatility adjustment on top. `SARIMAX.run()` therefore has a different
return signature (`model, test_y, preds` tuple) than other model scripts —
it's the only `run()` designed to be called from another model script rather
than only from `run_all.py`.

**Known gotcha:** `run_all.py`'s leaderboard collection only appends a
model's result if `mod.run()` returns a dict containing `"rmse"`
(`report()` produces such a dict, but almost no `run()` function currently
returns it back to the caller — most just call `report()` for its printed
side effect and return `None`). If you're debugging why
`imgs/model_family_comparison.png` is missing models, this is why — the fix
is to have each `run()` return what `report()` gives it.

**Data refresh caveat:** `data_loader.py` re-downloads from Yahoo Finance on
every run rather than using a fixed snapshot, so results (RMSE numbers,
ORDER constants from auto_arima) will drift over time as more history
accumulates.
