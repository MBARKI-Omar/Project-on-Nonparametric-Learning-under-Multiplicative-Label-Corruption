import numpy as np
from typing import Union, Tuple

class LabelNoiseEstimator:
    """
    Nonparametric estimator for the noise level η(x) in a multiplicative 
    label corruption model.
    
    This class implements the Nadaraya-Watson estimator using a Gaussian kernel 
    to approximate the regression function of observed labels Z given features X.
    """

    def __init__(self, h: float = 0.3):
        """
        Initializes the estimator with a specific bandwidth.

        Args:
            h (float): Bandwidth parameter (smoothing hyperparameter). 
                       Controls the bias-variance tradeoff.
        """
        self.h = h
        self.X_train = None
        self.Z_train = None

    def _gaussian_kernel(self, u: np.ndarray) -> np.ndarray:
        """
        Computes the Gaussian kernel K(u) = (1/sqrt(2π)) * exp(-u^2 / 2).
        
        Args:
            u (np.ndarray): Normalized distances between points.
            
        Returns:
            np.ndarray: Kernel weights.
        """
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

    def fit(self, X: np.ndarray, Z: np.ndarray):
        """
        Stores the dataset for local estimation.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            Z (np.ndarray): Observed labels vector of shape (n_samples,) in {-1, 1}.
        """
        self.X_train = X
        self.Z_train = Z

    def predict_regression(self, x_query: np.ndarray) -> float:
        """
        Estimates the regression function r(x) = E[Z|X=x] using 
        the Nadaraya-Watson formula.

        Args:
            x_query (np.ndarray): A single input point of shape (d,).

        Returns:
            float: Estimated value of r(x).
        """
        # Calculate Euclidean distances normalized by bandwidth h
        # For d-dimensional space: ||X_i - x|| / h
        diff = (self.X_train - x_query) / self.h
        distances = np.linalg.norm(diff, axis=1)

        # Compute weights via the kernel function
        weights = self._gaussian_kernel(distances)
        
        # Handle the denominator for the NW estimator
        sum_weights = np.sum(weights)
        
        if sum_weights == 0:
            # If no points are within reach of the kernel, return neutral prediction
            return 0.0
            
        return np.sum(weights * self.Z_train) / sum_weights

    def predict_eta(self, X_test: np.ndarray) -> np.ndarray:
        """
        Estimates the noise level η(X) based on the identifiability 
        formula derived in Part I:
        
        η(x) = (1 - |E[Z|X=x]|) / 2
        
        This assumes perfect class separability: |E[Y|X]| = 1.

        Args:
            X_test (np.ndarray): Points where η(x) should be estimated.

        Returns:
            np.ndarray: Estimated noise values in [0, 0.5[.
        """
        eta_hat = []
        for x in X_test:
            # Step 1: Estimate E[Z|X=x]
            r_hat = self.predict_regression(x)
            
            # Step 2: Apply the identifiability transformation
            e = (1 - np.abs(r_hat)) / 2
            eta_hat.append(e)
            
        # Clip values to ensure they remain in the theoretical range [0, 0.5[
        return np.clip(np.array(eta_hat), 0, 0.499)