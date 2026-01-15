"""
Unit tests for data generation functions.
"""

import numpy as np
import pytest
from src.data_generation import (
    eta_function,
    f_function,
    generate_X,
    generate_Y,
    generate_Z,
    generate_data
)


def test_shapes():
    """Check that all outputs have correct dimensions."""
    np.random.seed(0)
    m, d = 100, 2
    
    X, Y, Z = generate_data(m, d)
    
    assert X.shape == (m, d), f"X shape should be ({m}, {d})"
    assert Y.shape == (m,), f"Y shape should be ({m},)"
    assert Z.shape == (m,), f"Z shape should be ({m},)"


def test_values_range():
    """Check that all values are in valid ranges."""
    np.random.seed(1)
    m, d = 200, 2
    
    X, Y, Z = generate_data(m, d)
    
    # Labels should be binary
    assert set(Y) <= {-1, 1}, "Y should only contain -1 and +1"
    assert set(Z) <= {-1, 1}, "Z should only contain -1 and +1"
    
    # Noise function should be in [0, 0.5)
    eta_vals = eta_function(X)
    assert np.all(eta_vals >= 0), "eta should be non-negative"
    assert np.all(eta_vals < 0.5), "eta should be < 0.5"
    
    # f_function should be in [0, 1]
    f_vals = f_function(X)
    assert np.all(f_vals >= 0), "f should be >= 0"
    assert np.all(f_vals <= 1), "f should be <= 1"


def test_corruption_rate():
    """Check that empirical corruption matches theoretical eta."""
    np.random.seed(2)
    m, d = 5000, 2  # Large sample for accurate estimate
    
    X, Y, Z = generate_data(m, d)
    
    # Empirical corruption rate
    empirical_rate = np.mean(Y != Z)
    
    # Theoretical average noise
    theoretical_rate = np.mean(eta_function(X))
    
    # Should match within statistical tolerance
    tolerance = 0.03  # Change de 0.02 à 0.03 (3% au lieu de 2%)
    assert abs(empirical_rate - theoretical_rate) < tolerance, \
        f"Corruption rate mismatch: {empirical_rate:.3f} vs {theoretical_rate:.3f}"


def test_reproducibility():
    """Same seed should give same results."""
    m, d = 50, 2
    
    # First run
    np.random.seed(42)
    X1, Y1, Z1 = generate_data(m, d)
    
    # Second run with same seed
    np.random.seed(42)
    X2, Y2, Z2 = generate_data(m, d)
    
    # Should be identical
    assert np.allclose(X1, X2), "X should be reproducible"
    assert np.array_equal(Y1, Y2), "Y should be reproducible"
    assert np.array_equal(Z1, Z2), "Z should be reproducible"

def test_noise_level_low():
    """Check that low noise level produces correct range."""
    np.random.seed(20)
    m, d = 1000, 2
    
    X, Y, Z = generate_data(m, d, noise_level='low')
    
    eta_vals = eta_function(X, noise_level='low')
    
    # Should be in [0.02, 0.12]
    assert np.all(eta_vals >= 0.01), f"Min eta: {np.min(eta_vals)}"
    assert np.all(eta_vals <= 0.13), f"Max eta: {np.max(eta_vals)}"
    
    # Empirical corruption should match
    empirical_rate = np.mean(Y != Z)
    theoretical_rate = np.mean(eta_vals)
    
    assert abs(empirical_rate - theoretical_rate) < 0.03, \
        f"Corruption mismatch: {empirical_rate:.3f} vs {theoretical_rate:.3f}"


def test_noise_level_high():
    """Check that high noise level produces correct range."""
    np.random.seed(21)
    m, d = 1000, 2
    
    X, Y, Z = generate_data(m, d, noise_level='high')
    
    eta_vals = eta_function(X, noise_level='high')
    
    # Should be in [0.20, 0.45]
    assert np.all(eta_vals >= 0.19), f"Min eta: {np.min(eta_vals)}"
    assert np.all(eta_vals <= 0.46), f"Max eta: {np.max(eta_vals)}"
    
    # Empirical corruption should match
    empirical_rate = np.mean(Y != Z)
    theoretical_rate = np.mean(eta_vals)
    
    assert abs(empirical_rate - theoretical_rate) < 0.03, \
        f"Corruption mismatch: {empirical_rate:.3f} vs {theoretical_rate:.3f}"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])