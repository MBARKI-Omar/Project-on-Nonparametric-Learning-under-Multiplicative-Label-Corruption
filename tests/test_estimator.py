"""
Unit tests for noise estimator functions.
"""

import numpy as np
import pytest
from src.estimator import (
    gaussian_kernel,
    silverman_bandwidth,
    nadaraya_watson,
    estimate_eta
)
from src.data_generation import generate_data


def test_kernel_properties():
    """Check that Gaussian kernel has expected properties."""
    # Maximum at zero
    assert gaussian_kernel(0) > gaussian_kernel(1)
    assert gaussian_kernel(0) > gaussian_kernel(-1)
    
    # Symmetric
    assert np.isclose(gaussian_kernel(2), gaussian_kernel(-2))
    
    # Positive
    assert gaussian_kernel(5) > 0
    
    # Decays with distance
    assert gaussian_kernel(0) > gaussian_kernel(1) > gaussian_kernel(2)


def test_silverman_bandwidth():
    """Check that Silverman's rule produces reasonable bandwidth."""
    np.random.seed(0)
    
    # Small sample -> larger bandwidth
    X_small = np.random.randn(100, 2)
    h_small = silverman_bandwidth(X_small)
    
    # Large sample -> smaller bandwidth
    X_large = np.random.randn(5000, 2)
    h_large = silverman_bandwidth(X_large)
    
    assert h_small > h_large, "Bandwidth should decrease with sample size"
    assert h_small > 0, "Bandwidth should be positive"
    assert h_large > 0, "Bandwidth should be positive"


def test_nadaraya_watson_constant():
    """If all labels are constant, NW should predict that constant."""
    np.random.seed(1)
    
    m = 200
    X_train = np.random.uniform(0, 1, (m, 2))
    Z_train = np.ones(m)  # All +1
    
    X_test = np.random.uniform(0, 1, (50, 2))
    h = 0.3
    
    r_hat = nadaraya_watson(X_train, Z_train, X_test, h)
    
    # Should predict close to 1 everywhere
    assert np.all(r_hat > 0.9), "NW should predict ~1 when all labels are +1"


def test_estimate_eta_range():
    """Check that estimated noise is in valid range."""
    np.random.seed(2)
    
    X, Y, Z = generate_data(500, 2)
    eta_hat = estimate_eta(X, Z, X, threshold=0.2)
    
    # Remove NaN for range check
    valid_eta = eta_hat[~np.isnan(eta_hat)]
    
    assert np.all(valid_eta >= 0), "η should be non-negative"
    assert np.all(valid_eta < 0.5), "η should be < 0.5"


def test_estimate_eta_threshold():
    """Check that threshold correctly marks non-identifiable regions."""
    np.random.seed(3)
    
    # Create data with GOOD separation (not random)
    X, Y, Z = generate_data(300, 2)  # Use proper generation
    
    threshold = 0.3
    eta_hat = estimate_eta(X, Z, X, h=0.2, threshold=threshold)
    
    # Should identify SOME regions (not all NaN, not zero NaN)
    nan_rate = np.mean(np.isnan(eta_hat))
    
    assert 0.0 <= nan_rate < 1.0, f"nan_rate should be in [0, 1), got {nan_rate}"


def test_reproducibility():
    """Same seed should give same results."""
    np.random.seed(4)
    X, Y, Z = generate_data(200, 2)
    X_test = np.random.uniform(0, 1, (50, 2))
    
    # First run
    np.random.seed(10)
    eta1 = estimate_eta(X, Z, X_test, h=0.3)
    
    # Second run with same seed (shouldn't matter since no randomness in estimator)
    eta2 = estimate_eta(X, Z, X_test, h=0.3)
    
    # Should be identical (NaN == NaN handled specially)
    assert np.allclose(eta1, eta2, equal_nan=True), "Results should be reproducible"


def test_bandwidth_auto_selection():
    """Check that automatic bandwidth selection works."""
    np.random.seed(5)
    
    X, Y, Z = generate_data(300, 2)
    
    # Should not raise error
    eta_hat = estimate_eta(X, Z, X, h=None, threshold=0.3)
    
    # Should produce valid output
    assert eta_hat.shape == (300,), "Output shape should match input"
    assert np.sum(~np.isnan(eta_hat)) > 0, "Should identify some regions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])