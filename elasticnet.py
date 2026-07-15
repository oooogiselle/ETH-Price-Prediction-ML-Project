"""elasticnet.py -- Linear Regression with combined L1+L2 (ElasticNet) penalty on lag features."""

from sklearn.linear_model import ElasticNetCV
from preprocessing import load_clean_data, make_supervised_features, train_test_split_series
from utils import report, plot_predictions


def run():
    df = load_clean_data()
    feats = make_supervised_features(df, lags=10)
    train, test = train_test_split_series(feats)

    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    model = ElasticNetCV(cv=5, random_state=42).fit(X_train, y_train)
    preds = model.predict(X_test)

    report("ElasticNet", y_test, preds)
    plot_predictions(y_test, preds, "ElasticNet Regression - ETH-USD", "imgs/elasticnet_results.png")


if __name__ == "__main__":
    run()
