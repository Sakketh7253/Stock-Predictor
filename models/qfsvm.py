import numpy as np
from sklearn.svm import SVC
from quantum_kernel.q_kernel import QuantumKernelManager

class QFSVM_Model:
    """
    Quantum Fuzzy Support Vector Machine.
    Uses precomputed quantum kernel matrix and fuzzy logic sample weights for SVM.
    """
    def __init__(self, C=1.0):
        # We use precomputed kernel since we supply the quantum kernel matrix manually
        self.model = SVC(kernel='precomputed', C=C, probability=True)
        self.q_manager = None
        self.X_train_stored = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sample_weights: np.ndarray = None):
        """
        Trains the Quantum Fuzzy SVM using a precomputed quantum kernel matrix 
        and fuzzy sample weights.
        """
        feature_dim = X_train.shape[1]
        self.q_manager = QuantumKernelManager(feature_dimension=feature_dim)
        
        # Compute Kernel Matrix for training
        print("QFSVM: Precomputing Quantum Kernel matrix for training data...")
        K_train = self.q_manager.get_kernel_matrix(X_train)
        
        self.X_train_stored = X_train
        
        print(f"QFSVM: Training with {K_train.shape} kernel matrix...")
        self.model.fit(K_train, y_train, sample_weight=sample_weights)
        print("QFSVM: Training completed.")
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.X_train_stored is None:
            raise ValueError("Model must be trained before calling predict.")
            
        print("QFSVM: Precomputing Quantum Kernel matrix for test data predictions...")
        K_test = self.q_manager.get_kernel_matrix(X_test, self.X_train_stored)
        
        return self.model.predict(K_test)
