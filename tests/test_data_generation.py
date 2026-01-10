"""
test_data_generation.py

Test script for the data generation module.
"""

import numpy as np
from src.data_generation import generate_data, generate_X, generate_Y, generate_Z, eta_function

# Set global random seed for reproducibility
np.random.seed(42)


def test_basic_generation():
    """Test basic data generation with linear noise."""
    print("="*60)
    print("TEST 1: Basic Generation (linear noise)")
    print("="*60)
    
    X, Y, Z, eta_true = generate_data(n=100, d=1, noise_type='linear')
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Z shape: {Z.shape}")
    print(f"\nFirst 5 observations:")
    print(f"  X: {X[:5, 0]}")
    print(f"  Y: {Y[:5]}")
    print(f"  Z: {Z[:5]}")
    print(f"\nCorruption rate: {np.mean(Y != Z):.2%}")
    print(f"Expected corruption rate: {eta_true.mean():.2%}")
    print(f"η min: {eta_true.min():.3f}, η max: {eta_true.max():.3f}")
    print()


def test_noise_types():
    """Test different noise types."""
    print("="*60)
    print("TEST 2: Different Noise Types")
    print("="*60)
    
    noise_types = ['linear', 'sine']
    
    for noise_type in noise_types:
        print(f"\n{noise_type.upper()} noise:")
        X, Y, Z, eta_true = generate_data(n=500, d=1, noise_type=noise_type)
        
        print(f"  Corruption rate: {np.mean(Y != Z):.2%}")
        print(f"  η mean: {eta_true.mean():.3f}")
        print(f"  η min: {eta_true.min():.3f}, η max: {eta_true.max():.3f}")
    print()


def test_dimensions():
    """Test different dimensions."""
    print("="*60)
    print("TEST 3: Different Dimensions")
    print("="*60)
    
    dimensions = [1, 2, 5, 10]
    
    for d in dimensions:
        X, Y, Z, eta_true = generate_data(n=100, d=d, noise_type='linear')
        
        print(f"d={d:2d}: X shape = {X.shape}, corruption rate = {np.mean(Y != Z):.2%}")
    print()


def test_individual_functions():
    """Test individual generation functions."""
    print("="*60)
    print("TEST 4: Individual Functions")
    print("="*60)
    
    n, d = 10, 2
    
    # Test generate_X
    print("Testing generate_X:")
    X = generate_X(n=n, d=d)
    print(f"  X shape: {X.shape}")
    print(f"  X mean: {X.mean(axis=0)}")
    print(f"  X std: {X.std(axis=0)}")
    
    # Test generate_Y
    print("\nTesting generate_Y:")
    Y = generate_Y(X)
    print(f"  Y shape: {Y.shape}")
    print(f"  Y values: {np.unique(Y)}")
    print(f"  Class balance: +1: {np.mean(Y == 1):.1%}, -1: {np.mean(Y == -1):.1%}")
    
    # Test generate_Z
    print("\nTesting generate_Z:")
    Z, eta_values = generate_Z(X, Y, noise_type='linear')
    print(f"  Z shape: {Z.shape}")
    print(f"  Flipped labels: {np.sum(Y != Z)}/{n}")
    print(f"  η varies (not constant): {not np.all(eta_values == eta_values[0])}")
    
    # Test eta_function
    print("\nTesting eta_function:")
    eta_linear = eta_function(X, noise_type='linear')
    eta_sine = eta_function(X, noise_type='sine')
    print(f"  Linear η range: [{eta_linear.min():.3f}, {eta_linear.max():.3f}]")
    print(f"  Sine η range: [{eta_sine.min():.3f}, {eta_sine.max():.3f}]")
    print()


def test_reproducibility():
    """Test that global random seed was set correctly at the start."""
    print("="*60)
    print("TEST 5: Reproducibility Check")
    print("="*60)
    
    # Since we set np.random.seed(42) at the top of this file,
    # all random operations are reproducible
    
    # Generate some data
    X1, Y1, Z1, eta1 = generate_data(n=50, d=2)
    
    print(f"Data generated with global seed=42")
    print(f"First X value: {X1[0, 0]:.6f}")
    print(f"First Y value: {Y1[0]}")
    print(f"First Z value: {Z1[0]}")
    print(f"\nNote: Running this script multiple times will produce")
    print(f"      the same results because of the global seed.")
    print()


def test_corruption_statistics():
    """Test that corruption rate matches expected η."""
    print("="*60)
    print("TEST 6: Corruption Rate vs Expected η")
    print("="*60)
    
    n_large = 10000  # Large sample for better statistics
    
    print("LINEAR noise:")
    X, Y, Z, eta_true = generate_data(n=n_large, d=1, noise_type='linear')
    observed_rate = np.mean(Y != Z)
    expected_rate = eta_true.mean()
    print(f"  Expected η (mean): {expected_rate:.3f}")
    print(f"  Observed corruption: {observed_rate:.3f}")
    print(f"  Difference: {abs(observed_rate - expected_rate):.3f}")
    
    print("\nSINE noise:")
    X, Y, Z, eta_true = generate_data(n=n_large, d=1, noise_type='sine')
    observed_rate = np.mean(Y != Z)
    expected_rate = eta_true.mean()
    print(f"  Expected η (mean): {expected_rate:.3f}")
    print(f"  Observed corruption: {observed_rate:.3f}")
    print(f"  Difference: {abs(observed_rate - expected_rate):.3f}")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING DATA GENERATION TESTS")
    print("="*60 + "\n")
    
    test_basic_generation()
    test_noise_types()
    test_dimensions()
    test_individual_functions()
    test_reproducibility()
    test_corruption_statistics()
    
    print("="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)