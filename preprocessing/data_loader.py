import yfinance as yf
import pandas as pd
import os
import sys

# Add project root to sys.path to run this script standalone or as module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RAW_DATA_DIR, START_DATE, END_DATE

def load_data(ticker: str, start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance.
    Saves the data to data/raw/ticker.csv and returns the DataFrame.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}.")
        
    # Flatten MultiIndex columns if necessary (yfinance sometimes returns multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Save raw data
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}_raw.csv")
    df.to_csv(file_path)
    print(f"Raw data saved to {file_path}")
    
    return df

if __name__ == "__main__":
    # Test script standalone
    df = load_data("AAPL")
    print(df.head())
