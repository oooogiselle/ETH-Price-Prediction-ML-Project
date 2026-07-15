"""
random_forest.py -- NEW vs. the original BTC project.

Fits a RandomForestRegressor on lag features. Tree ensembles can capture
non-linear relationships the linear/ARIMA-family models miss, and don't
require stationarity assumptions.
"""

from sklearn.ensemble import RandomForestRegressor
from preprocessing import load_clean_data, make_supervised_features, train_test_split_series
from utils import report, plot_predictions


def run():
    df = load_clean_data()
    feats = make_supervised_features(df, lags=10)
    train, test = train_test_split_series(feats)

    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    ).fit(X_train, y_train)
    preds = model.predict(X_test)

    result = report("RandomForest", y_test, preds)
    plot_predictions(y_test, preds, "Random Forest Regression - ETH-USD", "imgs/random_forest_results.png")

    importances = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: -x[1])
    print("Top feature importances:", importances[:5])
    return result


if __name__ == "__main__":
    run()
