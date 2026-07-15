"""
run_all.py -- runs every model script and prints a final RMSE leaderboard,
mirroring the "which model wins" comparison from the original project's paper.
"""

import importlib
from visualize import plot_model_family_comparison

MODEL_MODULES = [
    "AR", "ARMA", "ARIMA", "SARIMAX", "GARCH_SARIMAX", "var_model",
    "elasticnet", "bayesian", "polyreg",
    "random_forest", "xgboost_model", "lstm_model",
]


def main():
    results = []
    for name in MODEL_MODULES:
        print(f"\n{'=' * 20} Running {name} {'=' * 20}")
        try:
            mod = importlib.import_module(name)
            result = mod.run()
            if isinstance(result, dict) and "rmse" in result:
                results.append(result)
        except Exception as e:
            print(f"  !! {name} failed: {e}")

    if results:
        plot_model_family_comparison(results)

    print("\nDone. Check the printed RMSE/MAE/MAPE lines above and the imgs/ folder for plots.")


if __name__ == "__main__":
    main()