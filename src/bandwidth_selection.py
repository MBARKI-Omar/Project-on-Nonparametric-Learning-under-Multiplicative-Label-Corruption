"""
src/bandwidth_selection.py

Implements Leave-One-Out Cross-Validation (LOOCV) for optimal bandwidth selection.
Uses the 'Hat Matrix' algebraic shortcut (PRESS statistic) to avoid retraining.
"""

import numpy as np

class BandwidthSelector:
    def __init__(self, h_grid=None):
        """
        Args:
            h_grid: List or array of bandwidths to test. 
                    If None, generates a default logarithmic range.
        """
        if h_grid is None:
            # Recherche logarithmique pour couvrir des échelles différentes
            self.h_grid = np.logspace(np.log10(0.05), np.log10(2.0), 30)
        else:
            self.h_grid = h_grid

    def _gaussian_kernel(self, u):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

    def loocv_score(self, h, X, Z):
        """
        Computes the LOOCV error for a specific bandwidth h
        using the algebraic shortcut: Mean of [ (Z_i - r_hat(X_i)) / (1 - S_ii) ]^2
        """
        n = len(X)
        squared_errors = 0.0
        
        # Optimization: broadcasting allows computing this without nested loops for reasonable n
        # For very large n, a KDTree would be better, but this fits the project scope.
        
        # 1. Compute Distance Matrix (n x n)
        # Reshape for broadcasting: (n, 1, d) - (1, n, d) -> (n, n, d)
        diff = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) / h
        dists_sq = np.sum(diff**2, axis=2) # Squared Euclidean distance
        
        # 2. Compute Kernel Matrix
        # K(u) = exp(-0.5 * u^2) / sqrt(2pi)
        # We ignore the constant factor for weights normalization as it cancels out
        W = np.exp(-0.5 * dists_sq)
        
        # 3. Compute Diagonal Elements S_ii (Self-influence)
        # S_ii = W_ii / sum(W_i)
        # W_ii is exp(0) = 1
        sum_W = np.sum(W, axis=1)
        S_ii = 1.0 / (sum_W + 1e-10) # 1.0 comes from W[i,i]
        
        # 4. Compute Predictions (Nadaraya-Watson)
        # r_hat = W * Z / sum_W
        # We need (W @ Z) element-wise
        # Z needs shape (n,)
        numerator = W @ Z
        r_hat = numerator / (sum_W + 1e-10)
        
        # 5. Compute LOOCV Residuals using the Magic Formula
        # residual = (Z_i - r_hat_i) / (1 - S_ii)
        # Filter unstable points where S_ii is close to 1 (isolated points)
        valid_points = S_ii < 0.999
        
        residuals = np.zeros(n)
        if np.any(valid_points):
            residuals[valid_points] = (Z[valid_points] - r_hat[valid_points]) / (1 - S_ii[valid_points])
            
        return np.mean(residuals[valid_points]**2)

    def select_optimal_h(self, X, Z):
        """
        Grid search to find the bandwidth that minimizes the LOOCV score.
        """
        best_h = None
        min_error = float('inf')
        
        for h in self.h_grid:
            try:
                error = self.loocv_score(h, X, Z)
                if error < min_error:
                    min_error = error
                    best_h = h
            except Exception:
                continue
                
        return best_h, min_error