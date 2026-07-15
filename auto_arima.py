"""
auto_arima.py -- runs pmdarima's automated grid search to find good (p,d,q)
order for ETH close prices. The best order found here is what AR.py, ARMA.py,
ARIMA.py and SARIMAX.py use.
"""

import pmdarima as pm
from preprocessing import load_clean_data, train_test_split_series, TARGET_COL


def find_best_order():
    df = load_clean_data()
    train, _ = train_test_split_series(df[TARGET_COL])

    model = pm.auto_arima(
        train,
        start_p=0, start_q=0, max_p=5, max_q=5,
        d=None, seasonal=False,
        trace=True, error_action="ignore", suppress_warnings=True,
        stepwise=True,
    )
    print(model.summary())
    return model.order


if __name__ == "__main__":
    order = find_best_order()
    print(f"\nBest (p,d,q) order found: {order}")
