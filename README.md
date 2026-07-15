# ETH-Price-Prediction-ML-Project

A time-series & ML project predicting **Ethereum (ETH-USD)** prices, adapted
from [dushyant18033/BTC-Price-Prediction-ML-Project](https://github.com/dushyant18033/BTC-Price-Prediction-ML-Project).

## What's different from the original BTC project

| | Original (BTC) | This variation (ETH) |
|---|---|---|
| Asset | Bitcoin | Ethereum |
| Data source | CoinMarketCap web API | `yfinance` (free, no API key) |
| Classical models | AR, ARMA, ARIMA, SARIMAX, GARCH-SARIMAX, VAR | same suite, kept for comparison |
| Regression models | ElasticNet, Bayesian Ridge, Polynomial Regression | same suite, kept for comparison |
| **New models** | — | **Random Forest, XGBoost, LSTM** |
| Model comparison | Manual, via paper | `run_all.py` runs everything and prints an RMSE/MAE/MAPE leaderboard |

The classical statistical baseline is kept intact so the new tree-based and
deep-learning models can be benchmarked directly against it on the same
data and the same train/test split.

## Project Structure

```
data_loader.py       # pulls ETH-USD OHLCV data via yfinance
preprocessing.py      # cleaning, ADF stationarity test, differencing, lag features
utils.py              # RMSE/MAE/MAPE metrics + plotting helpers

auto_arima.py          # pmdarima grid search for best (p,d,q)
AR.py / ARMA.py / ARIMA.py / SARIMAX.py / GARCH_SARIMAX.py / var_model.py

elasticnet.py / bayesian.py / polyreg.py   # classical regression on lag features

random_forest.py       # NEW: tree ensemble
xgboost_model.py        # NEW: gradient boosting
lstm_model.py            # NEW: recurrent neural net on windowed sequences

run_all.py              # runs every model, prints leaderboard
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python data_loader.py     # fetch and cache ETH-USD data
python auto_arima.py      # (optional) find best ARIMA order, update ORDER constants
python run_all.py         # run every model and print the RMSE leaderboard
```

Or run any single model script directly, e.g. `python xgboost_model.py`.

## Models Used

- **Classical time series**: AR, ARMA, ARIMA, SARIMAX, GARCH+SARIMAX (volatility-adjusted), VAR
- **Classical regression**: ElasticNet, Bayesian Ridge, Polynomial Regression
- **Tree ensembles (new)**: Random Forest, XGBoost
- **Deep learning (new)**: LSTM

## Notes

- Data is fetched fresh from Yahoo Finance every time `data_loader.py` runs, so
  results will drift as more history accumulates -- unlike the original
  project's fixed CoinMarketCap snapshot.
- The train/test split is chronological (last 15% held out), never shuffled,
  since this is a time series.
