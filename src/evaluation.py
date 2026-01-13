"""
evaluation.py

Metrics for evaluating noise function estimation performance.
"""

import numpy as np
from typing import Dict, Optional


def mse(eta_true: np.ndarray, eta_hat: np.ndarray) -> float:
    """
    Compute Mean Squared Error between true and estimated noise.
    
    Args:
        eta_true: True noise values of shape (n,)
        eta_hat: Estimated noise values of shape (n,)
        
    Returns:
        MSE value
    """
    return np.mean((eta_true - eta_hat) ** 2)


def mae(eta_true: np.ndarray, eta_hat: np.ndarray) -> float:
    """
    Compute Mean Absolute Error between true and estimated noise.
    
    Args:
        eta_true: True noise values of shape (n,)
        eta_hat: Estimated noise values of shape (n,)
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(eta_true - eta_hat))


def max_error(eta_true: np.ndarray, eta_hat: np.ndarray) -> float:
    """
    Compute maximum absolute error (worst case).
    
    Args:
        eta_true: True noise values of shape (n,)
        eta_hat: Estimated noise values of shape (n,)
        
    Returns:
        Maximum absolute error
    """
    return np.max(np.abs(eta_true - eta_hat))


def rmse(eta_true: np.ndarray, eta_hat: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        eta_true: True noise values of shape (n,)
        eta_hat: Estimated noise values of shape (n,)
        
    Returns:
        RMSE value
    """
    return np.sqrt(mse(eta_true, eta_hat))


def l2_norm(eta_true: np.ndarray, eta_hat: np.ndarray) -> float:
    """
    Compute L2 norm of the error.
    
    Args:
        eta_true: True noise values of shape (n,)
        eta_hat: Estimated noise values of shape (n,)
        
    Returns:
        L2 norm of error
    """
    return np.linalg.norm(eta_true - eta_hat)


def relative_error(eta_true: np.ndarray, eta_hat: np.ndarray, 
                   epsilon: float = 1e-10) -> float:
    """
    Compute relative error (normalized by true values).
    
    Args:
        eta_true: True noise values of shape (n,)
        eta_hat: Estimated noise values of shape (n,)
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Mean relative error
    """
    return np.mean(np.abs(eta_true - eta_hat) / (np.abs(eta_true) + epsilon))


def pointwise_errors(eta_true: np.ndarray, eta_hat: np.ndarray) -> np.ndarray:
    """
    Compute pointwise absolute errors.
    
    Args:
        eta_true: True noise values of shape (n,)
        eta_hat: Estimated noise values of shape (n,)
        
    Returns:
        Array of pointwise errors of shape (n,)
    """
    return np.abs(eta_true - eta_hat)


def evaluate_estimation(eta_true: np.ndarray, 
                       eta_hat: np.ndarray,
                       X_test: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        eta_true: True noise values of shape (n,)
        eta_hat: Estimated noise values of shape (n,)
        X_test: Test points (optional, for additional metrics)
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'mse': mse(eta_true, eta_hat),
        'mae': mae(eta_true, eta_hat),
        'rmse': rmse(eta_true, eta_hat),
        'max_error': max_error(eta_true, eta_hat),
        'l2_norm': l2_norm(eta_true, eta_hat),
        'relative_error': relative_error(eta_true, eta_hat)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    
    for key, value in metrics.items():
        print(f"  {key:20s}: {value:.6f}")
    
    print("=" * 60)


def aggregate_metrics(metrics_list: list) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from multiple trials (Monte Carlo).
    
    Args:
        metrics_list: List of metric dictionaries from multiple trials
        
    Returns:
        Dictionary with mean and std for each metric
    """
    if not metrics_list:
        return {}
    
    # Get all metric names
    metric_names = metrics_list[0].keys()
    
    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in metrics_list]
        aggregated[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    return aggregated


def print_aggregated_metrics(aggregated: Dict[str, Dict[str, float]], 
                            title: str = "Aggregated Metrics"):
    """
    Pretty print aggregated metrics.
    
    Args:
        aggregated: Dictionary of aggregated metrics
        title: Title to display
    """
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'Metric':<20s} {'Mean':<12s} {'Std':<12s} {'Min':<12s} {'Max':<12s}")
    print("-" * 70)
    
    for metric_name, stats in aggregated.items():
        print(f"{metric_name:<20s} {stats['mean']:<12.6f} {stats['std']:<12.6f} "
              f"{stats['min']:<12.6f} {stats['max']:<12.6f}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test the evaluation module
    print("Testing evaluation module...\n")
    
    # Generate synthetic test data
    np.random.seed(42)
    n = 100
    eta_true = np.random.uniform(0.1, 0.4, n)
    eta_hat = eta_true + np.random.randn(n) * 0.05  # Add some noise
    
    # Compute metrics
    metrics = evaluate_estimation(eta_true, eta_hat)
    print_metrics(metrics, "Single Trial Metrics")
    
    # Test aggregation
    print("\n\nTesting metric aggregation...\n")
    metrics_list = []
    for _ in range(10):
        eta_hat_trial = eta_true + np.random.randn(n) * 0.05
        metrics_list.append(evaluate_estimation(eta_true, eta_hat_trial))
    
    aggregated = aggregate_metrics(metrics_list)
    print_aggregated_metrics(aggregated, "Aggregated Metrics (10 trials)")