"""polyreg.py -- Linear Regression augmented with polynomial features."""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from preprocessing import load_clean_data, make_supervised_features, train_test_split_series
from utils import report, plot_predictions

DEGREE = 2


def run():
    df = load_clean_data()
    feats = make_supervised_features(df, lags=10)
    train, test = train_test_split_series(feats)

    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    model = make_pipeline(PolynomialFeatures(degree=DEGREE), LinearRegression())
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result = report("PolynomialRegression", y_test, preds)
    plot_predictions(y_test, preds, "Polynomial Regression - ETH-USD", "imgs/polyreg_results.png")
    return result


if __name__ == "__main__":
    run()
