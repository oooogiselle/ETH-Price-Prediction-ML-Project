"""
lstm_model.py -- NEW vs. the original BTC project.

Fits a small LSTM recurrent network on scaled, windowed closing-price
sequences. Included as a deep-learning comparison point against the
classical statistical models and the tree ensembles -- the original
project used no neural methods at all.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from preprocessing import load_clean_data, train_test_split_series, TARGET_COL
from utils import report, plot_predictions

WINDOW = 30


def make_windows(series, window=WINDOW):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def run():
    df = load_clean_data()
    train_s, test_s = train_test_split_series(df[TARGET_COL])

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_s.values.reshape(-1, 1)).flatten()
    # include the tail of train so the first test predictions have full windows
    full_test_input = np.concatenate([train_scaled[-WINDOW:], scaler.transform(test_s.values.reshape(-1, 1)).flatten()])

    X_train, y_train = make_windows(train_scaled)
    X_test, y_test_scaled = make_windows(full_test_input)

    X_train = X_train.reshape((*X_train.shape, 1))
    X_test = X_test.reshape((*X_test.shape, 1))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    result = report("LSTM", test_s.values, preds)
    plot_predictions(test_s, preds, "LSTM Model - ETH-USD", "imgs/lstm_results.png")
    return result


if __name__ == "__main__":
    run()
