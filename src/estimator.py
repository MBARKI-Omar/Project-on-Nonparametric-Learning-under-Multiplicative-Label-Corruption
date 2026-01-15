"""
Nonparametric noise estimation using Nadaraya-Watson kernel regression.
Implements the method from Part II of the project.
"""

import numpy as np


def gaussian_kernel(u):
    """
    Gaussian kernel function.
    
    Args:
        u: array of any shape - distances
    
    Returns:
        K(u) = exp(-u²/2) / sqrt(2π)
    """
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def silverman_bandwidth(X):
    """
    Calculate optimal bandwidth using Silverman's rule of thumb.
    
    Formula: h = (4/(d+2))^(1/(d+4)) * σ * m^(-1/(d+4))
    where σ is the average std dev across dimensions
    
    Args:
        X: array of shape (m, d)
    
    Returns:
        h: scalar bandwidth
    """
    m, d = X.shape
    sigma = np.mean(np.std(X, axis=0))
    
    h = ((4.0 / (d + 2)) ** (1.0 / (d + 4))) * sigma * (m ** (-1.0 / (d + 4)))
    
    return h


def nadaraya_watson(X_train, Z_train, X_test, h):
    """
    Nadaraya-Watson kernel regression estimator.
    Estimates E[Z|X=x] for each test point.
    
    Args:
        X_train: training features, shape (m, d)
        Z_train: training labels, shape (m,)
        X_test: test points, shape (n, d)
        h: bandwidth (can be None for auto-selection)
    
    Returns:
        r_hat: predictions, shape (n,)
    """
    # Auto-select bandwidth if needed
    if h is None:
        h = silverman_bandwidth(X_train)
    
    m_test = X_test.shape[0]
    r_hat = np.zeros(m_test)
    
    for i in range(m_test):
        # Compute distances to all training points
        distances = np.linalg.norm(X_train - X_test[i], axis=1)
        
        # Compute kernel weights
        weights = gaussian_kernel(distances / h)
        
        # Weighted average
        if np.sum(weights) > 0:
            r_hat[i] = np.sum(weights * Z_train) / np.sum(weights)
        else:
            # If no points nearby, fallback to empirical mean
            r_hat[i] = np.mean(Z_train)
    
    return r_hat


def estimate_eta(X_train, Z_train, X_test, h=None, threshold=0.3):
    """
    Estimate noise function η(x) using local separability assumption.
    
    Method:
    1. Estimate r(x) = E[Z|X=x] using Nadaraya-Watson
    2. Apply formula: η(x) = (1 - |r(x)|) / 2
    3. Mark points with |r(x)| < threshold as non-identifiable (NaN)
    
    Args:
        X_train: training features, shape (m, d)
        Z_train: noisy labels, shape (m,)
        X_test: test points, shape (n, d)
        h: bandwidth (optional, uses Silverman if None)
        threshold: confidence threshold τ (default 0.3)
    
    Returns:
        eta_hat: estimated noise, shape (n,)
                 Contains NaN for non-identifiable regions
    
    Example:
        >>> X, Y, Z = generate_data(1000, 2)
        >>> eta_hat = estimate_eta(X, Z, X, threshold=0.3)
        >>> identifiable = ~np.isnan(eta_hat)
        >>> print(f"Identifiable: {np.mean(identifiable):.1%}")
    """
    # Auto-select bandwidth if not provided
    if h is None:
        h = silverman_bandwidth(X_train)
    
    # Step 1: Estimate E[Z|X] using NW
    r_hat = nadaraya_watson(X_train, Z_train, X_test, h)
    
    # Step 2: Apply inversion formula
    eta_hat = (1.0 - np.abs(r_hat)) / 2.0
    
    # Step 3: Mark non-identifiable regions
    # Where |r_hat| < threshold, we have weak separation -> set to NaN
    non_identifiable = np.abs(r_hat) < threshold
    eta_hat[non_identifiable] = np.nan
    
    return eta_hat