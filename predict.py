"""
predict.py - Standalone Prediction Script for QFSVM Stock Market Predictor

Runs the full pipeline (Data Download → Preprocessing → Feature Engineering
→ Fuzzy Logic → Model Training → Evaluation) and prints a human-readable 
prediction report for all models.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import AVAILABLE_TICKERS, START_DATE, END_DATE
from pipeline.train_pipeline import TrainingPipeline

# ─── Configuration ────────────────────────────────────────────────────────────
TICKER      = "AAPL"       # Change to any of: AAPL, TSLA, GOOG, RELIANCE.NS, TCS.NS
START       = "2020-01-01"
END         = "2024-01-01"

# Which models to run? Remove any you want to skip (QFSVM is slow – quantum sim)
MODELS_TO_RUN = ["RF", "SVM", "LSTM"]   # add "QFSVM" when you want quantum kernel
# ──────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {1: "📈 Uptrend", 0: "➡️  Neutral", -1: "📉 Downtrend"}

def print_header(text: str, width: int = 60):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def print_metrics(name: str, metrics: dict):
    print(f"\n  ┌─ {name} ─{'─' * max(0, 40 - len(name))}┐")
    for k, v in metrics.items():
        print(f"  │  {k:<18}: {v:.4f}")
    print(f"  └{'─' * 43}┘")

def print_predictions(y_true: np.ndarray, y_pred: np.ndarray, dates, n: int = 10):
    """Print last n predictions vs actuals."""
    print(f"\n  Last {n} Predictions:")
    print(f"  {'Date':<14} {'Actual':<18} {'Predicted':<18} {'Match'}")
    print(f"  {'-'*60}")
    for i in range(-min(n, len(y_true)), 0):
        actual    = LABEL_MAP.get(y_true[i], str(y_true[i]))
        predicted = LABEL_MAP.get(y_pred[i], str(y_pred[i]))
        match     = "✅" if y_true[i] == y_pred[i] else "❌"
        try:
            date = str(dates[i])[:10]
        except Exception:
            date = "N/A"
        print(f"  {date:<14} {actual:<18} {predicted:<18} {match}")

def main():
    print_header(f"QFSVM Stock Market Predictor — {TICKER}")
    print(f"  Date Range : {START}  →  {END}")
    print(f"  Models     : {', '.join(MODELS_TO_RUN)}")

    pipeline    = TrainingPipeline()
    all_metrics = {}

    for model_type in MODELS_TO_RUN:
        print_header(f"Running Model: {model_type}")
        try:
            results = pipeline.run_pipeline(
                ticker     = TICKER,
                start_date = START,
                end_date   = END,
                model_type = model_type,
            )

            y_true        = results["y_test"]
            y_pred        = results["y_pred"]
            metrics       = results["metrics"]
            test_indices  = results["test_indices"]

            all_metrics[model_type] = metrics
            print_metrics(model_type, metrics)
            print_predictions(y_true, y_pred, test_indices)

        except Exception as exc:
            print(f"\n  ⚠️  {model_type} failed: {exc}")

    # ── Summary Comparison ────────────────────────────────────────────────────
    if all_metrics:
        print_header("Model Comparison Summary")
        key_metrics = ["Accuracy", "F1 Score", "RMSE"]
        header = f"  {'Model':<20}" + "".join(f"{m:<14}" for m in key_metrics)
        print(header)
        print("  " + "-" * (20 + 14 * len(key_metrics)))
        for model, m in all_metrics.items():
            row = f"  {model:<20}" + "".join(f"{m.get(k, 0.0):<14.4f}" for k in key_metrics)
            print(row)

        best_model = max(all_metrics, key=lambda m: all_metrics[m]["Accuracy"])
        print(f"\n  🏆 Best Model by Accuracy: {best_model} "
              f"({all_metrics[best_model]['Accuracy']:.4f})")

    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
