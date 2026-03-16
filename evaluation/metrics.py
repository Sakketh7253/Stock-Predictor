import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error

class ModelEvaluator:
    def __init__(self):
        pass
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculates and returns classification and regression metrics.
        Target values expected: -1, 0, 1.
        """
        metrics = {
            'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'Recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'F1 Score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': float(mean_absolute_error(y_true, y_pred))
        }
        return metrics
        
    def print_metrics(self, model_name: str, metrics: dict):
        print(f"\n--- {model_name} Evaluation ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("-" * 30)
