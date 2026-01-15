"""
Evaluation metrics for noise estimation.
"""

import numpy as np
from src.estimator import estimate_eta


def compute_mse(eta_true, eta_pred):
    """
    Mean Squared Error on identifiable points only.
    
    Args:
        eta_true: true noise values, shape (n,)
        eta_pred: predicted noise values, shape (n,), may contain NaN
    
    Returns:
        mse: float, MSE computed on non-NaN points
    """
    # Filter out NaN predictions
    mask = ~np.isnan(eta_pred)
    
    if np.sum(mask) == 0:
        return np.nan  # No identifiable points
    
    return np.mean((eta_true[mask] - eta_pred[mask]) ** 2)


def compute_mae(eta_true, eta_pred):
    """
    Mean Absolute Error on identifiable points only.
    
    Args:
        eta_true: true noise values, shape (n,)
        eta_pred: predicted noise values, shape (n,), may contain NaN
    
    Returns:
        mae: float, MAE computed on non-NaN points
    """
    mask = ~np.isnan(eta_pred)
    
    if np.sum(mask) == 0:
        return np.nan
    
    return np.mean(np.abs(eta_true[mask] - eta_pred[mask]))


def compute_rmse(eta_true, eta_pred):
    """
    Root Mean Squared Error on identifiable points only.
    Same unit as eta, more interpretable than MSE.
    
    Args:
        eta_true: true noise values, shape (n,)
        eta_pred: predicted noise values, shape (n,), may contain NaN
    
    Returns:
        rmse: float, sqrt(MSE)
    """
    mse = compute_mse(eta_true, eta_pred)
    return np.sqrt(mse) if not np.isnan(mse) else np.nan


def compute_max_error(eta_true, eta_pred):
    """
    Maximum absolute error on identifiable points.
    Useful for worst-case analysis.
    
    Args:
        eta_true: true noise values, shape (n,)
        eta_pred: predicted noise values, shape (n,), may contain NaN
    
    Returns:
        max_error: float, max |eta_true - eta_pred|
    """
    mask = ~np.isnan(eta_pred)
    
    if np.sum(mask) == 0:
        return np.nan
    
    return np.max(np.abs(eta_true[mask] - eta_pred[mask]))


def compute_identifiable_rate(eta_pred):
    """
    Compute percentage of identifiable points.
    
    Args:
        eta_pred: predicted noise values, shape (n,), may contain NaN
    
    Returns:
        rate: float in [0, 1], percentage of non-NaN points
    """
    return np.mean(~np.isnan(eta_pred))


def evaluate_estimator(X_train, Z_train, X_test, eta_true_func, h=None, threshold=0.3):
    """
    Complete evaluation of the estimator.
    
    Args:
        X_train: training features, shape (m, d)
        Z_train: noisy labels, shape (m,)
        X_test: test points, shape (n, d)
        eta_true_func: function that computes true eta given X
        h: bandwidth (optional)
        threshold: confidence threshold
    
    Returns:
        metrics: dict with keys 'mse', 'mae', 'rmse', 'max_error', 'identifiable_rate'
    
    Example:
        >>> from src.data_generation import generate_data, eta_function
        >>> X, Y, Z = generate_data(1000, 2)
        >>> metrics = evaluate_estimator(X, Z, X, eta_function)
        >>> print(f"MSE: {metrics['mse']:.4f}")
    """
    # Estimate noise
    eta_pred = estimate_eta(X_train, Z_train, X_test, h=h, threshold=threshold)
    
    # Compute true noise at test points
    eta_true = eta_true_func(X_test)
    
    # Compute all metrics
    metrics = {
        'mse': compute_mse(eta_true, eta_pred),
        'mae': compute_mae(eta_true, eta_pred),
        'rmse': compute_rmse(eta_true, eta_pred),
        'max_error': compute_max_error(eta_true, eta_pred),
        'identifiable_rate': compute_identifiable_rate(eta_pred)
    }
    
    return metrics