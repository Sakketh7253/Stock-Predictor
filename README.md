# Stock Market Forecasting Based on Quantum Fuzzy Support Vector Machine (QFSVM)

This research-level project implements a hybrid machine learning model to predict stock market trends. The primary model, **Quantum Fuzzy Support Vector Machine (QFSVM)**, integrates Fuzzy Logic for handling financial data uncertainty and a Quantum-Inspired Kernel (ZZFeatureMap via Qiskit simulator) for optimal non-linear feature mapping. 

The system provides performance comparisons against traditional support vector machines (SVM), Random Forest, and Long Short-Term Memory (LSTM) neural networks.

## Features

- **Data Engineering**: Automated extraction of historical stock prices (`yfinance`) and calculation of technical indicators (`ta`).
- **Fuzzy Logic Component**: Generates membership weights to reduce the impact of noisy market data.
- **Quantum Feature Map**: Classical stock features are mapped into a simulated quantum Hilbert space using Qiskit.
- **Model Baselines**: Includes traditional traditional ML options for robust evaluation.
- **Streamlit Dashboard**: An interactive, user-friendly interface.

## Directory Structure
```
Stock Market Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── config/
│   └── config.py
├── preprocessing/
│   ├── data_loader.py
│   └── preprocessor.py
├── feature_engineering/
│   └── technical_indicators.py
├── fuzzy_logic/
│   └── fuzzy_layer.py
├── quantum_kernel/
│   └── q_kernel.py
├── models/
│   ├── qfsvm.py
│   ├── svm_model.py
│   ├── rf_model.py
│   └── lstm_model.py
├── pipeline/
│   └── train_pipeline.py
├── evaluation/
│   └── metrics.py
├── visualization/
│   └── plots.py
├── app/
│   └── dashboard.py
├── main.py
├── requirements.txt
└── README.md
```

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Application**
   ```bash
   streamlit run main.py
   ```
