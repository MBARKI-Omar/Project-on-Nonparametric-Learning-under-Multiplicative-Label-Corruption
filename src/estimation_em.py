"""
src/estimation_em.py

Implementation of the Non-Parametric EM algorithm with Dual Bandwidths.
Using separate bandwidths for signal and noise allows for much better estimation
when the signal is sharp (decision boundary) and the noise is smooth.
"""

import numpy as np

class EMNoiseEstimator:
    """
    Non-parametric EM estimator with Dual Bandwidth capability.
    
    Attributes:
        h_signal (float): Bandwidth for estimating the decision boundary f(x).
                          Should be small to capture sharp transitions.
        h_noise (float): Bandwidth for estimating the noise function eta(x).
                         Should be large to smooth out local variance.
    """

    def __init__(self, h_signal=0.2, h_noise=1.0, max_iter=50, tol=1e-4):
        self.h_signal = h_signal
        self.h_noise = h_noise  # Souvent plus grand que h_signal
        self.max_iter = max_iter
        self.tol = tol
        self.X_train = None
        self.Z_train = None
        self.eta_map = None 

    def _gaussian_kernel(self, dists):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * dists**2)

    def _weighted_nadaraya_watson(self, X_source, Y_source, X_target, h, weights=None):
        """Generic NW regression with specific bandwidth h."""
        n_target = len(X_target)
        preds = np.zeros(n_target)
        
        for i in range(n_target):
            # Use the specific h passed as argument
            diff = (X_source - X_target[i]) / h
            dists = np.linalg.norm(diff, axis=1)
            
            k_weights = self._gaussian_kernel(dists)
            
            if weights is not None:
                final_weights = k_weights * weights
            else:
                final_weights = k_weights
                
            sum_w = np.sum(final_weights)
            
            if sum_w > 1e-10:
                preds[i] = np.sum(final_weights * Y_source) / sum_w
            else:
                preds[i] = np.mean(Y_source)
                
        return preds

    def fit(self, X, Z):
        self.X_train = X
        Z_01 = (Z + 1) / 2
        n = len(X)
        
        # --- INITIALIZATION ---
        # Initialize noise with a prior (slightly higher to encourage detection)
        eta_curr = np.full(n, 0.25)
        
        # Initialize Signal f(x) using h_signal
        f_curr = self._weighted_nadaraya_watson(X, Z_01, X, h=self.h_signal)
        
        print(f"Starting EM (h_signal={self.h_signal}, h_noise={self.h_noise})...")
        
        for iteration in range(self.max_iter):
            prev_eta = eta_curr.copy()
            
            # --- E-STEP ---
            prob_z1 = f_curr * (1 - eta_curr) + (1 - f_curr) * eta_curr
            gamma = np.zeros(n)
            epsilon = 1e-10
            
            mask_z1 = (Z_01 == 1)
            gamma[mask_z1] = ((1 - eta_curr[mask_z1]) * f_curr[mask_z1]) / (prob_z1[mask_z1] + epsilon)
            
            mask_z0 = (Z_01 == 0)
            prob_z0 = 1 - prob_z1
            gamma[mask_z0] = (eta_curr[mask_z0] * f_curr[mask_z0]) / (prob_z0[mask_z0] + epsilon)
            
            # --- M-STEP ---
            
            # 1. Update Signal f(x) using SMALL bandwidth (h_signal)
            # We want to fit the decision boundary closely
            f_next = self._weighted_nadaraya_watson(X, gamma, X, h=self.h_signal)
            
            # 2. Update Noise eta(x) using LARGE bandwidth (h_noise)
            # We want to smooth the noise estimate over a larger area
            local_error_prob = np.abs(Z_01 - gamma)
            eta_next = self._weighted_nadaraya_watson(X, local_error_prob, X, h=self.h_noise)
            
            eta_next = np.clip(eta_next, 0.0, 0.499)
            
            # Check convergence
            diff = np.mean(np.abs(eta_next - prev_eta))
            if diff < self.tol:
                print(f"-> Converged at iter {iteration+1}")
                self.eta_map = eta_next
                break
            
            eta_curr = eta_next
            f_curr = f_next
            
            if iteration == self.max_iter - 1:
                self.eta_map = eta_next

    def predict_eta(self, X_test):
        if self.eta_map is None:
            raise ValueError("Not fitted")
        # Propagate noise estimate using the SMOOTH bandwidth (h_noise)
        return self._weighted_nadaraya_watson(self.X_train, self.eta_map, X_test, h=self.h_noise)