"""
QFSVM Stock Market Forecasting - Dashboard
Simple, clean Streamlit interface.
"""
import os
import sys
import warnings
import datetime
import traceback

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import AVAILABLE_TICKERS
from pipeline.train_pipeline import TrainingPipeline
from visualization.plots import Visualizer

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("📈 Stock Market Forecasting")
st.markdown("**Hybrid model combining Fuzzy Logic + Quantum Kernel + Support Vector Machine**")
st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

ticker     = st.sidebar.selectbox("Stock Ticker", AVAILABLE_TICKERS)
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date   = st.sidebar.date_input("End Date",   datetime.date(2024, 1, 1))

MODEL_MAP = {
    "Random Forest (RF)":          "RF",
    "Support Vector Machine (SVM)": "SVM",
    "LSTM Neural Network":          "LSTM",
    "QFSVM (Quantum + Fuzzy)":      "QFSVM",
}
model_choice = st.sidebar.selectbox("Model", list(MODEL_MAP.keys()))
model_type   = MODEL_MAP[model_choice]

if model_type == "QFSVM":
    st.sidebar.warning(
        "⚠️ **QFSVM** uses a quantum circuit simulator. "
        "Training is capped at 100 samples and may take **1–3 minutes**."
    )

run_btn = st.sidebar.button("🚀 Train & Predict")

# ─── Session State ────────────────────────────────────────────────────────────
if "results_store" not in st.session_state:
    st.session_state.results_store = {}

# ─── Run Pipeline ─────────────────────────────────────────────────────────────
if run_btn:
    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.stop()

    with st.spinner(f"Training **{model_choice}** on **{ticker}** — please wait…"):
        try:
            results = TrainingPipeline().run_pipeline(
                ticker=ticker,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                model_type=model_type,
            )
            run_key = f"{ticker}_{model_type}_{start_date}_{end_date}"
            st.session_state.results_store[run_key] = {
                "results": results,
                "ticker":  ticker,
                "model":   model_choice,
            }
            st.success("✅ Training complete!")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc(), language="python")
            st.stop()

# ─── Results ──────────────────────────────────────────────────────────────────
if st.session_state.results_store:
    entry      = list(st.session_state.results_store.values())[-1]
    results    = entry["results"]
    vis_ticker = entry["ticker"]
    vis_model  = entry["model"]

    df         = results["df"]
    metrics    = results["metrics"]
    y_test     = results["y_test"]
    y_pred     = results["y_pred"]
    test_dates = results["test_indices"]

    # ── Metrics ──
    st.subheader(f"📋 Evaluation — {vis_model} on {vis_ticker}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{metrics.get('Accuracy',  0):.4f}")
    col2.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
    col3.metric("Recall",    f"{metrics.get('Recall',    0):.4f}")
    col4.metric("F1 Score",  f"{metrics.get('F1 Score',  0):.4f}")

    col5, col6, _, __ = st.columns(4)
    col5.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
    col6.metric("MAE",  f"{metrics.get('MAE',  0):.4f}")

    # ── Download ──
    pred_df = pd.DataFrame({
        "Date":      list(test_dates),
        "Actual":    y_test,
        "Predicted": y_pred,
        "Correct":   (np.array(y_test) == np.array(y_pred)).astype(int),
    })
    st.download_button(
        "⬇️ Download Predictions (CSV)",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{vis_ticker}_{vis_model.replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Charts ──
    Visualizer.plot_stock_trend(df, vis_ticker)
    st.markdown("---")
    Visualizer.plot_predictions(y_test, y_pred, test_dates)
    st.markdown("---")
    Visualizer.plot_correlation_heatmap(df)
    st.markdown("---")
    Visualizer.plot_metrics_comparison({vis_model: metrics})

    with st.expander("🗃️ View Processed Data (last 25 rows)"):
        show_cols = [c for c in df.columns if "_Fuzzy_" not in c]
        st.dataframe(df[show_cols].tail(25), use_container_width=True)

# ─── Landing ──────────────────────────────────────────────────────────────────
else:
    st.info("👈 Configure the settings in the sidebar and click **Train & Predict** to start.")

    st.markdown("""
    ### How it works

    | Step | Description |
    |------|-------------|
    | 1️⃣ **Data** | Downloads historical OHLCV data from Yahoo Finance |
    | 2️⃣ **Preprocessing** | Missing values, noise filtering, IQR outlier capping |
    | 3️⃣ **Indicators** | Computes MA, EMA, RSI, MACD, Bollinger Bands, ATR |
    | 4️⃣ **Fuzzy Logic** | Assigns certainty weights to reduce noisy-data impact |
    | 5️⃣ **Quantum Kernel** | Maps features into quantum Hilbert space (ZZFeatureMap) |
    | 6️⃣ **QFSVM** | Trains SVM with quantum kernel + fuzzy sample weights |
    | 7️⃣ **Evaluation** | Accuracy, F1, RMSE, MAE vs actual Uptrend/Neutral/Downtrend |
    """)
