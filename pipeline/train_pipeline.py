import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import TEST_SIZE, RANDOM_STATE
from preprocessing.data_loader import load_data
from preprocessing.preprocessor import DataPreprocessor
from feature_engineering.technical_indicators import TechnicalIndicatorEngineer
from fuzzy_logic.fuzzy_layer import FuzzyLayer
from evaluation.metrics import ModelEvaluator

from models.qfsvm import QFSVM_Model
from models.svm_model import BaselineSVM
from models.rf_model import BaselineRF
from models.lstm_model import BaselineLSTM

class TrainingPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = TechnicalIndicatorEngineer()
        self.fuzzy_layer = FuzzyLayer()
        self.evaluator = ModelEvaluator()
        
    def run_pipeline(self, ticker: str, start_date: str, end_date: str, model_type: str = 'QFSVM'):
        """
        Executes the entire end-to-end pipeline for the selected ticker and model.
        Available models: 'QFSVM', 'SVM', 'RF', 'LSTM'
        """
        print(f"\n[Pipeline] Starting for {ticker} using {model_type}...")
        
        # 1. Load Data
        df_raw = load_data(ticker, start_date, end_date)
        
        # 2. Preprocess
        df_processed = self.preprocessor.process(df_raw, ticker)
        
        # 3. Feature Engineering
        df_features = self.feature_engineer.add_indicators(df_processed)
        
        # Features to be used for modeling
        feature_cols = ['MA_14', 'EMA_14', 'RSI_14', 'MACD', 'MACD_Signal', 
                        'MACD_Diff', 'BB_High', 'BB_Low', 'BB_Mid', 
                        'Momentum_14', 'ATR_14']
        
        # Preserve original price columns BEFORE scaling (used for visualization)
        viz_cols = ['Close', 'MA_14', 'BB_High', 'BB_Low', 'Smoothed_Close']
        for col in viz_cols:
            if col in df_features.columns:
                df_features[f'{col}_Orig'] = df_features[col].copy()

        # Scale features
        df_features = self.preprocessor.scale_features(df_features, feature_cols)
        
        # 4. Fuzzy Logic
        df_fuzzy = self.fuzzy_layer.apply_fuzzy_weights(df_features, feature_cols)
        
        X = df_fuzzy[feature_cols].values
        y = df_fuzzy['Trend'].values
        weights = df_fuzzy['Fuzzy_Certainty_Weight'].values
        
        # QFSVM is computationally intensive on simulators. Limit to 100 samples to avoid timeout.
        if model_type == 'QFSVM' and len(X) > 100:
            print("[QFSVM] Limiting dataset size to 100 samples (quantum simulation constraint).")
            X = X[-100:]
            y = y[-100:]
            weights = weights[-100:]
            df_fuzzy = df_fuzzy.iloc[-100:]
            
        # Optional: reduce feature dimension for Quantum Kernel to avoid deep circuits
        if model_type == 'QFSVM' and X.shape[1] > 4:
            from sklearn.decomposition import PCA
            print("[QFSVM] Applying PCA to reduce features to 4 for ZZFeatureMap.")
            pca = PCA(n_components=4)
            X = pca.fit_transform(X)
            
        # 5. Train / Test Split
        X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
            X, y, weights, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
        ) # Time series mapping relies on shuffle=False
        
        # 6. Model Training and Prediction
        print(f"Training Model ({model_type})...")
        if model_type == 'QFSVM':
            model = QFSVM_Model(C=1.0)
            model.fit(X_train, y_train, sample_weights=w_train)
        elif model_type == 'SVM':
            model = BaselineSVM()
            model.fit(X_train, y_train)
        elif model_type == 'RF':
            model = BaselineRF()
            model.fit(X_train, y_train)
        elif model_type == 'LSTM':
            model = BaselineLSTM(input_shape=(1, X_train.shape[1]), num_classes=3)
            model.fit(X_train, y_train, epochs=20, batch_size=16)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        print("Generating Predictions...")
        y_pred = model.predict(X_test)
        
        # 7. Evaluation
        metrics = self.evaluator.evaluate(y_test, y_pred)
        self.evaluator.print_metrics(model_type, metrics)
        
        return {
            'model': model,
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'df': df_fuzzy,
            'test_indices': df_fuzzy.index[-len(y_test):] # Date indices for plotting
        }

if __name__ == "__main__":
    from config.config import START_DATE, END_DATE
    pipeline = TrainingPipeline()
    # Test quickly with RF
    results = pipeline.run_pipeline("AAPL", "2023-01-01", "2024-01-01", model_type="RF")
