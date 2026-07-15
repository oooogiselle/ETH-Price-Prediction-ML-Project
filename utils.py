"""
utils.py -- shared metrics + plotting helpers used across all model scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def report(model_name, y_true, y_pred):
    r, m, p = rmse(y_true, y_pred), mae(y_true, y_pred), mape(y_true, y_pred)
    print(f"[{model_name}] RMSE={r:.2f}  MAE={m:.2f}  MAPE={p:.2f}%")
    return {"model": model_name, "rmse": r, "mae": m, "mape": p}


def plot_predictions(y_true, y_pred, title, out_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.index if hasattr(y_true, "index") else range(len(y_true)), y_true, label="Actual")
    plt.plot(y_true.index if hasattr(y_true, "index") else range(len(y_true)), y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("ETH-USD Close Price")
    plt.legend()
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()
    plt.close()
