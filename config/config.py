import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Ensure Data Directories Exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Default Data Configurations
TICKER = "AAPL"
AVAILABLE_TICKERS = ["AAPL", "TSLA", "GOOG", "RELIANCE.NS", "TCS.NS"]
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"

# Data Preprocessing
TEST_SIZE = 0.2
RANDOM_STATE = 42

# QFSVM and Kernel Configuration
QISKIT_SHOTS = 1024
FUZZY_BINS = 3  # Low, Medium, High
