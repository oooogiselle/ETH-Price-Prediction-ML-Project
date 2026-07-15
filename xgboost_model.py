"""
xgboost_model.py -- NEW vs. the original BTC project.

Fits a gradient-boosted tree ensemble (XGBoost) on lag features. Usually
one of the strongest tabular-data baselines and a natural comparison
point against both the classical ARIMA family and RandomForest.
"""

from xgboost import XGBRegressor
from preprocessing import load_clean_data, make_supervised_features, train_test_split_series
from utils import report, plot_predictions


def run():
    df = load_clean_data()
    feats = make_supervised_features(df, lags=10)
    train, test = train_test_split_series(feats)

    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ).fit(X_train, y_train)
    preds = model.predict(X_test)

    report("XGBoost", y_test, preds)
    plot_predictions(y_test, preds, "XGBoost Regression - ETH-USD", "imgs/xgboost_results.png")


if __name__ == "__main__":
    run()
