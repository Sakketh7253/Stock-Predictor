import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class QuantumKernelManager:
    """
    Manages the translation of classical stock features into a quantum Hilbert space
    using the ZZFeatureMap and computes the quantum kernel matrix.
    """
    def __init__(self, feature_dimension: int, reps: int = 2):
        self.feature_dimension = feature_dimension
        self.reps = reps
        
        # Define the ZZFeatureMap (simulates mapping features into quantum states)
        self.feature_map = ZZFeatureMap(feature_dimension=self.feature_dimension, 
                                        reps=self.reps, 
                                        entanglement='linear')

        # Use StatevectorSampler (BaseSamplerV2) — required by newer qiskit_algorithms
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)

        # Create Quantum Kernel Simulator instance
        self.qkernel = FidelityQuantumKernel(feature_map=self.feature_map, fidelity=fidelity)
        
    def get_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Calculates the quantum kernel matrix utilizing local quantum simulator.
        If X2 is None, calculates the symmetric kernel matrix for X1.
        """
        print(f"Calculating Quantum Kernel Matrix (Features: {self.feature_dimension})...")
        if X2 is None:
            return self.qkernel.evaluate(x_vec=X1)
        else:
            return self.qkernel.evaluate(x_vec=X1, y_vec=X2)
