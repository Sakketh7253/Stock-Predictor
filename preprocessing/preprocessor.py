import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PROCESSED_DATA_DIR

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill then backward fill missing values."""
        df = df.ffill().bfill()
        return df
        
    def filter_noise_and_outliers(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Smooth noise using a moving average on the 'Close' column 
        and cap outliers using IQR.
        """
        # Smoothing Close price as 'Smoothed_Close'
        df['Smoothed_Close'] = df['Close'].rolling(window=window, min_periods=1).mean()
        
        # Outlier capping using IQR on Close price
        Q1 = df['Close'].quantile(0.25)
        Q3 = df['Close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df.loc[:, 'Close'] = np.clip(df['Close'], lower_bound, upper_bound)
        return df
        
    def scale_features(self, df: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
        """Scales specified features using StandardScaler."""
        df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        return df
        
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target labels: 1 for Uptrend, -1 for Downtrend, 0 for Neutral.
        Based on next day's close compared to today's close.
        Threshold represents neutral movement percentage (e.g., 0.1%).
        """
        threshold = 0.001 
        df['Daily_Return'] = df['Close'].pct_change()
        df['Target_Return'] = df['Daily_Return'].shift(-1)
        
        conditions = [
            (df['Target_Return'] > threshold),
            (df['Target_Return'] < -threshold)
        ]
        choices = [1, -1] # 1 Uptrend, -1 Downtrend
        
        df['Trend'] = np.select(conditions, choices, default=0) # 0 Neutral
        df = df.dropna(subset=['Target_Return']) # Drop the last row since we don't know the future
        
        return df
        
    def process(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        print("Preprocessing data...")
        df = self.handle_missing_values(df)
        df = self.filter_noise_and_outliers(df)
        df = self.create_labels(df)
        
        # Save preprocessed standard (non scaled features) into processed
        file_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_preprocessed.csv")
        df.to_csv(file_path)
        print(f"Preprocessed data saved to {file_path}")
        return df
        
if __name__ == "__main__":
    from data_loader import load_data
    raw_df = load_data("AAPL", end_date="2024-01-01")
    processor = DataPreprocessor()
    processed_df = processor.process(raw_df, "AAPL")
    print(processed_df.head())
