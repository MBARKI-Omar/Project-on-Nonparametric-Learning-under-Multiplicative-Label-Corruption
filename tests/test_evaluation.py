"""
Unit tests for evaluation metrics.
"""

import numpy as np
import pytest
from src.evaluation import (
    compute_mse,
    compute_mae,
    compute_rmse,
    compute_max_error,
    compute_identifiable_rate,
    evaluate_estimator
)
from src.data_generation import generate_data, eta_function


def test_compute_mse_with_nan():
    """Check that MSE correctly ignores NaN values."""
    eta_true = np.array([0.1, 0.2, 0.3, 0.4])
    eta_pred = np.array([0.12, np.nan, 0.28, 0.42])
    
    # Should compute MSE only on indices 0 and 2,3
    expected = np.mean([(0.12-0.1)**2, (0.28-0.3)**2, (0.42-0.4)**2])
    
    mse = compute_mse(eta_true, eta_pred)
    
    assert np.isclose(mse, expected), f"Expected {expected}, got {mse}"


def test_compute_mae_with_nan():
    """Check that MAE correctly ignores NaN values."""
    eta_true = np.array([0.1, 0.2, 0.3, 0.4])
    eta_pred = np.array([0.15, np.nan, 0.25, np.nan])
    
    # Should compute MAE only on indices 0 and 2
    expected = np.mean([0.05, 0.05])
    
    mae = compute_mae(eta_true, eta_pred)
    
    assert np.isclose(mae, expected), f"Expected {expected}, got {mae}"


def test_compute_rmse():
    """Check that RMSE is sqrt of MSE."""
    eta_true = np.array([0.1, 0.2, 0.3])
    eta_pred = np.array([0.11, 0.19, 0.32])
    
    mse = compute_mse(eta_true, eta_pred)
    rmse = compute_rmse(eta_true, eta_pred)
    
    assert np.isclose(rmse, np.sqrt(mse)), "RMSE should be sqrt(MSE)"


def test_compute_max_error():
    """Check that max error finds the largest deviation."""
    eta_true = np.array([0.1, 0.2, 0.3, 0.4])
    eta_pred = np.array([0.12, 0.25, np.nan, 0.35])
    
    # Max error should be at index 3: |0.4 - 0.35| = 0.05
    # (index 1 has |0.2 - 0.25| = 0.05, index 0 has 0.02)
    expected = 0.05
    
    max_err = compute_max_error(eta_true, eta_pred)
    
    assert np.isclose(max_err, expected), f"Expected {expected}, got {max_err}"


def test_identifiable_rate():
    """Check that identifiable rate counts non-NaN correctly."""
    eta_pred = np.array([0.1, np.nan, 0.3, np.nan, 0.5])
    
    # 3 out of 5 are identifiable
    expected = 3 / 5
    
    rate = compute_identifiable_rate(eta_pred)
    
    assert np.isclose(rate, expected), f"Expected {expected}, got {rate}"


def test_evaluate_estimator():
    """Check that evaluate_estimator returns all metrics."""
    np.random.seed(10)
    
    X, Y, Z = generate_data(500, 2)
    
    metrics = evaluate_estimator(X, Z, X, eta_function, threshold=0.3)
    
    # Check all keys present
    required_keys = ['mse', 'mae', 'rmse', 'max_error', 'identifiable_rate']
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"
    
    # Check values are reasonable
    assert metrics['mse'] > 0, "MSE should be positive"
    assert metrics['rmse'] > 0, "RMSE should be positive"
    assert 0 < metrics['identifiable_rate'] <= 1, "Rate should be in (0, 1]"


def test_all_nan_case():
    """Check behavior when all predictions are NaN."""
    eta_true = np.array([0.1, 0.2, 0.3])
    eta_pred = np.array([np.nan, np.nan, np.nan])
    
    # All metrics should return NaN
    assert np.isnan(compute_mse(eta_true, eta_pred))
    assert np.isnan(compute_mae(eta_true, eta_pred))
    assert np.isnan(compute_rmse(eta_true, eta_pred))
    assert np.isnan(compute_max_error(eta_true, eta_pred))
    
    # Identifiable rate should be 0
    assert compute_identifiable_rate(eta_pred) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])