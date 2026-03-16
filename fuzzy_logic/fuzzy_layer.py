import numpy as np
import pandas as pd
from typing import Tuple

class FuzzyLayer:
    """
    Implements a fuzzy layer to handle uncertainty in financial data.
    """
    def __init__(self):
        pass
        
    def _triangular_membership(self, x, a, b, c):
        """Standard triangular membership function."""
        return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))
        
    def _compute_fuzzy_memberships(self, feature_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Computes Low, Medium, High memberships for a given feature.
        Uses percentiles dynamically define the sets.
        """
        p_low = feature_series.quantile(0.25)
        p_med = feature_series.quantile(0.50)
        p_high = feature_series.quantile(0.75)
        
        min_val = feature_series.min()
        max_val = feature_series.max()
        
        # Left open triangular for Low
        low_mem = np.maximum(0, np.minimum(1, (p_med - feature_series) / (p_med - min_val + 1e-9)))
        
        # Triangular for Medium
        med_mem = self._triangular_membership(feature_series, p_low, p_med, p_high)
        
        # Right open triangular for High
        high_mem = np.maximum(0, np.minimum(1, (feature_series - p_med) / (max_val - p_med + 1e-9)))
        
        return low_mem, med_mem, high_mem

    def apply_fuzzy_weights(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Computes fuzzy memberships for key features and assigns an overall fuzzy weight
        to each sample. The weight acts to reduce the impact of noisy/uncertain samples.
        """
        df_fuzzy = df.copy()
        
        total_certainty = np.zeros(len(df))
        
        for feature in features:
            low, med, high = self._compute_fuzzy_memberships(df[feature])
            
            # Create Fuzzy Features (Low, Medium, High) for the dataset
            df_fuzzy[f'{feature}_Fuzzy_Low'] = low
            df_fuzzy[f'{feature}_Fuzzy_Med'] = med
            df_fuzzy[f'{feature}_Fuzzy_High'] = high
            
            # Simple fuzzy inference rule: samples near the median of indicators 
            # (e.g., normal volatility, neutral RSI) have higher certainty.
            # Extreme values (high uncertainty) receive lower weights.
            total_certainty += med
            
        # Normalize weights between 0 and 1
        df_fuzzy['Fuzzy_Certainty_Weight'] = total_certainty / len(features)
        
        return df_fuzzy
