"""
utils.py

Utility functions for experiments.
These wrappers ensure compatibility when changing Q1 or Q2.
"""

import numpy as np
import os
import json
import pandas as pd
from typing import Dict, Tuple, Optional

# Import core modules
from data_generation import generate_data, eta_function
from estimators import (
    estimate_eta_batch, 
    silverman_bandwidth, 
    cross_validation_bandwidth
)
from evaluation import evaluate_estimation


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def setup_experiment(config, create_dirs: bool = True):
    """
    Setup experiment environment.
    
    Args:
        config: ExperimentConfig object
        create_dirs: Whether to create output directories
    """
    # Set random seed
    np.random.seed(config.random_seed)
    
    # Create directories
    if create_dirs:
        os.makedirs(config.figures_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        print(f"Directories created: {config.figures_dir}, {config.results_dir}")


# ============================================================================
# DATA GENERATION WRAPPER
# ============================================================================

def generate_experiment_data(m: int, d: int, noise_type: str, 
                            seed: Optional[int] = None) -> Dict:
    """
    Generate data for experiments.
    
    This wrapper ensures compatibility when changing data_generation.py.
    
    Args:
        m: Sample size
        d: Dimension
        noise_type: Type of noise ('linear', 'sine')
        seed: Random seed (optional)
        
    Returns:
        Dictionary with keys 'X', 'Y', 'Z', 'eta_true'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Call data generation function
    # NOTE: If you change the signature of generate_data(), update here
    result = generate_data(n=m, d=d, noise_type=noise_type)
    
    # Normalize output to dictionary format
    if isinstance(result, tuple):
        return {
            'X': result[0],
            'Y': result[1],
            'Z': result[2],
            'eta_true': result[3]
        }
    else:
        # Already a dictionary
        return result


# ============================================================================
# ESTIMATION WRAPPER
# ============================================================================

def estimate_with_config(X_test: np.ndarray, 
                        X_train: np.ndarray,
                        Z_train: np.ndarray,
                        config,
                        h_method: Optional[str] = None,
                        h_value: Optional[float] = None) -> np.ndarray:
    """
    Estimate noise function using configuration settings.
    
    This wrapper ensures compatibility when changing estimators.py.
    
    Args:
        X_test: Test points of shape (n_test, d)
        X_train: Training points of shape (n_train, d)
        Z_train: Training labels of shape (n_train,)
        config: ExperimentConfig object
        h_method: Bandwidth selection method ('silverman', 'cv', or None to use config)
        h_value: Manual bandwidth value (overrides h_method if provided)
        
    Returns:
        Estimated noise values of shape (n_test,)
    """
    # Determine bandwidth
    if h_value is not None:
        # Manual bandwidth provided
        h = h_value
    else:
        # Select bandwidth based on method
        method = h_method if h_method is not None else config.bandwidth_method
        
        if method == 'silverman':
            h = silverman_bandwidth(X_train)
        elif method == 'cv':
            h = cross_validation_bandwidth(X_train, Z_train, config.h_grid, 
                                          kernel=config.default_kernel)
        else:
            h = config.default_bandwidth
    
    # Call estimation function
    # NOTE: If you change the signature of estimate_eta_batch(), update here
    eta_hat = estimate_eta_batch(
        X_test=X_test,
        X_train=X_train,
        Z_train=Z_train,
        h=h,
        kernel=config.default_kernel
    )
    
    return eta_hat


# ============================================================================
# EVALUATION WRAPPER
# ============================================================================

def evaluate_with_config(data_dict: Dict, 
                        eta_hat: np.ndarray,
                        X_test: np.ndarray,
                        noise_type: str) -> Dict:
    """
    Evaluate estimation and compute metrics.
    
    Args:
        data_dict: Data dictionary from generate_experiment_data
        eta_hat: Estimated noise values
        X_test: Test points where eta_hat was computed
        noise_type: Type of noise (for computing true eta at test points)
        
    Returns:
        Dictionary of metrics
    """
    # Compute true eta at test points
    eta_true_test = eta_function(X_test, noise_type=noise_type)
    
    # Evaluate
    metrics = evaluate_estimation(eta_true_test, eta_hat, X_test)
    
    return metrics


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results_csv(data: Dict, filename: str, config):
    """
    Save results to CSV file.
    
    Args:
        data: Dictionary of results
        filename: Output filename (without extension)
        config: ExperimentConfig object
    """
    filepath = os.path.join(config.results_dir, f"{filename}.csv")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
    print(f"Results saved: {filepath}")


def save_results_json(data: Dict, filename: str, config):
    """
    Save results to JSON file.
    
    Args:
        data: Dictionary of results
        filename: Output filename (without extension)
        config: ExperimentConfig object
    """
    filepath = os.path.join(config.results_dir, f"{filename}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    data_serializable = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data_serializable[key] = value.tolist()
        elif isinstance(value, dict):
            # Recursively handle nested dicts
            data_serializable[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                     for k, v in value.items()}
        else:
            data_serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(data_serializable, f, indent=2)
    
    print(f"Results saved: {filepath}")


# ============================================================================
# GRID GENERATION FOR EVALUATION
# ============================================================================

def generate_test_grid_1d(X_train: np.ndarray, n_points: int = 100,
                         margin: float = 0.2) -> np.ndarray:
    """
    Generate 1D grid for evaluation.
    
    Args:
        X_train: Training data of shape (n, d)
        n_points: Number of test points
        margin: Margin beyond data range (fraction of range)
        
    Returns:
        Test grid of shape (n_points, d)
    """
    x_min = X_train[:, 0].min()
    x_max = X_train[:, 0].max()
    x_range = x_max - x_min
    
    # Extend range with margin
    x_min_extended = x_min - margin * x_range
    x_max_extended = x_max + margin * x_range
    
    # Generate grid
    x_grid = np.linspace(x_min_extended, x_max_extended, n_points)
    
    # Match dimension of training data
    if X_train.shape[1] == 1:
        return x_grid.reshape(-1, 1)
    else:
        # For higher dimensions, create grid only on first dimension
        # and use median values for other dimensions
        X_test = np.zeros((n_points, X_train.shape[1]))
        X_test[:, 0] = x_grid
        for j in range(1, X_train.shape[1]):
            X_test[:, j] = np.median(X_train[:, j])
        return X_test


def generate_test_sample(X_train: np.ndarray, n_points: int = 50) -> np.ndarray:
    """
    Generate random test points from same distribution as training.
    
    Args:
        X_train: Training data of shape (n, d)
        n_points: Number of test points
        
    Returns:
        Test points of shape (n_points, d)
    """
    d = X_train.shape[1]
    
    # Generate from same distribution (assuming Gaussian)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_test = np.random.randn(n_points, d) * std + mean
    
    return X_test


# ============================================================================
# SUMMARY PRINTING
# ============================================================================

def print_experiment_summary(title: str, config_info: Dict, results: Dict):
    """
    Print experiment summary.
    
    Args:
        title: Experiment title
        config_info: Configuration parameters
        results: Results dictionary
    """
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)
    
    print("\nConfiguration:")
    for key, value in config_info.items():
        print(f"  {key:20s}: {value}")
    
    print("\nResults:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.6f}")
        else:
            print(f"  {key:20s}: {value}")
    
    print("="*70 + "\n")


# ============================================================================
# PROGRESS BAR (SIMPLE)
# ============================================================================

def print_progress(current: int, total: int, prefix: str = "Progress"):
    """
    Print simple progress bar.
    
    Args:
        current: Current iteration (0-indexed)
        total: Total iterations
        prefix: Prefix text
    """
    percent = 100 * (current + 1) / total
    filled = int(50 * (current + 1) // total)
    bar = '█' * filled + '-' * (50 - filled)
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current+1}/{total})', end='')
    if current == total - 1:
        print()  # New line at end


if __name__ == "__main__":
    # Test utils module
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    
    print("Testing utils module...\n")
    
    # Test setup
    setup_experiment(config)
    
    # Test data generation
    print("\n1. Testing data generation wrapper...")
    data = generate_experiment_data(m=100, d=1, noise_type='linear', seed=42)
    print(f"   Generated data with keys: {data.keys()}")
    print(f"   X shape: {data['X'].shape}")
    print(f"   Y shape: {data['Y'].shape}")
    print(f"   Z shape: {data['Z'].shape}")
    
    # Test estimation
    print("\n2. Testing estimation wrapper...")
    X_test = generate_test_grid_1d(data['X'], n_points=20)
    eta_hat = estimate_with_config(X_test, data['X'], data['Z'], config)
    print(f"   Estimated eta_hat shape: {eta_hat.shape}")
    print(f"   eta_hat range: [{eta_hat.min():.3f}, {eta_hat.max():.3f}]")
    
    # Test evaluation
    print("\n3. Testing evaluation wrapper...")
    metrics = evaluate_with_config(data, eta_hat, X_test, 'linear')
    print(f"   Metrics: {list(metrics.keys())}")
    print(f"   MSE: {metrics['mse']:.6f}")
    
    print("\n✓ All utils tests passed!")