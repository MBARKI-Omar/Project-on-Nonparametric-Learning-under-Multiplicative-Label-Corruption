"""
Data generation for label noise estimation project.
Creates synthetic datasets where we know the true noise function.
"""

import numpy as np


def eta_function(X):
    """
    True noise function: probability of label flip at each point.
    
    Args:
        X: array of shape (m, d) - feature vectors
    
    Returns:
        array of shape (m,) - noise rates in [0, 0.5)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if X.shape[1] == 1:
        # 1D: varies between 0.05 and 0.35
        return 0.2 + 0.15 * np.sin(np.pi * X[:, 0])
    else:
        # 2D+: adjust coefficients to stay positive
        # Range: [0.05, 0.35]
        return 0.2 + 0.08 * np.sin(2 * np.pi * X[:, 0]) + 0.07 * np.cos(2 * np.pi * X[:, 1])


def f_function(X):
    """
    True regression function: P(Y=1|X).
    Creates regions where one class dominates.
    
    Args:
        X: array of shape (m, d)
    
    Returns:
        array of shape (m,) - probabilities in [0, 1]
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if X.shape[1] == 1:
        # 1D: steep sigmoid creates clear separation
        z = 10 * (X[:, 0] - 0.5)
    else:
        # 2D+: linear boundary with steep transition
        z = 5 * (X[:, 0] + X[:, 1] - 1.0)
    
    return 1.0 / (1.0 + np.exp(-z))


def generate_X(m, d):
    """
    Generate random feature vectors.
    
    Args:
        m: number of samples
        d: dimension
    
    Returns:
        X: array of shape (m, d), uniform in [0, 1]^d
    """
    return np.random.uniform(0, 1, size=(m, d))


def generate_Y(X):
    """
    Generate true labels based on f_function.
    Each label is sampled from Bernoulli(f(X)).
    
    Args:
        X: array of shape (m, d)
    
    Returns:
        Y: array of shape (m,) with values in {-1, +1}
    """
    m = X.shape[0]
    probs = f_function(X)
    
    # Sample Y=+1 with probability f(X), else Y=-1
    return np.where(np.random.rand(m) < probs, 1, -1)


def generate_Z(X, Y):
    """
    Corrupt labels Y into observed labels Z.
    Each label flips with probability eta(X).
    
    Args:
        X: array of shape (m, d)
        Y: array of shape (m,) - true labels
    
    Returns:
        Z: array of shape (m,) - noisy labels in {-1, +1}
    """
    m = X.shape[0]
    noise_probs = eta_function(X)
    
    # Flip label with probability eta(X)
    flip = np.random.rand(m) < noise_probs
    return np.where(flip, -Y, Y)


def generate_data(m, d):
    """
    Generate complete synthetic dataset.
    
    Args:
        m: number of samples
        d: feature dimension
    
    Returns:
        X: features, shape (m, d)
        Y: true labels, shape (m,)
        Z: noisy labels, shape (m,)
    
    Example:
        >>> np.random.seed(42)
        >>> X, Y, Z = generate_data(1000, 2)
        >>> print(f"Corruption rate: {np.mean(Y != Z):.2%}")
    """
    X = generate_X(m, d)
    Y = generate_Y(X)
    Z = generate_Z(X, Y)
    
    return X, Y, Z