"""
Experiment 4: Robustness to decision boundaries.
Compare well-separated vs overlapping scenarios.
"""

import numpy as np
import json

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation import generate_data, eta_function
from src.estimator import estimate_eta, nadaraya_watson
from src.evaluation import compute_mse, compute_identifiable_rate
from src.visualization import plot_eta_2d


def f_function_separated(X, slope=10):
    """Well-separated classes (steep boundary)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    z = slope * (X[:, 0] + X[:, 1] - 1.0)
    return 1.0 / (1.0 + np.exp(-z))


def f_function_overlap(X, slope=2):
    """Overlapping classes (gentle boundary)."""
    return f_function_separated(X, slope=slope)


def generate_data_custom_f(m, d, f_func, noise_level='medium'):
    """Generate data with custom f function."""
    from src.data_generation import generate_X, eta_function, generate_Z
    
    X = generate_X(m, d)
    
    # Generate Y using custom f
    probs = f_func(X)
    Y = np.where(np.random.rand(m) < probs, 1, -1)
    
    # Generate Z
    Z = generate_Z(X, Y, noise_level=noise_level)
    
    return X, Y, Z


def run_experiment():
    """Run boundary robustness experiment."""
    print("=" * 60)
    print("Experiment 4: Boundary Robustness")
    print("=" * 60)
    
    np.random.seed(42)
    
    m, d = 2000, 2
    grid_size = 50
    
    # Scenario A: Well-separated
    print(f"\nScenario A: Well-separated classes (m={m}, d={d})")
    X_sep, Y_sep, Z_sep = generate_data_custom_f(m, d, 
                                                  lambda x: f_function_separated(x, slope=10),
                                                  noise_level='medium')
    
    # Scenario B: Overlapping
    print(f"Scenario B: Overlapping classes (m={m}, d={d})")
    X_ovl, Y_ovl, Z_ovl = generate_data_custom_f(m, d,
                                                  lambda x: f_function_overlap(x, slope=2),
                                                  noise_level='medium')
    
    # Create grid for visualization
    x1 = np.linspace(0, 1, grid_size)
    x2 = np.linspace(0, 1, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Evaluate both scenarios
    print("\nEstimating noise on grid...")
    
    # Scenario A
    eta_true_grid_sep = eta_function(X_grid).reshape(grid_size, grid_size)
    eta_pred_flat_sep = estimate_eta(X_sep, Z_sep, X_grid, threshold=0.3)
    eta_pred_grid_sep = eta_pred_flat_sep.reshape(grid_size, grid_size)
    r_hat_flat_sep = nadaraya_watson(X_sep, Z_sep, X_grid, h=None)
    r_hat_grid_sep = r_hat_flat_sep.reshape(grid_size, grid_size)
    
    # Scenario B
    eta_true_grid_ovl = eta_function(X_grid).reshape(grid_size, grid_size)
    eta_pred_flat_ovl = estimate_eta(X_ovl, Z_ovl, X_grid, threshold=0.3)
    eta_pred_grid_ovl = eta_pred_flat_ovl.reshape(grid_size, grid_size)
    r_hat_flat_ovl = nadaraya_watson(X_ovl, Z_ovl, X_grid, h=None)
    r_hat_grid_ovl = r_hat_flat_ovl.reshape(grid_size, grid_size)
    
    # Compute metrics
    mse_sep = compute_mse(eta_true_grid_sep.ravel(), eta_pred_flat_sep)
    rate_sep = compute_identifiable_rate(eta_pred_flat_sep)
    
    mse_ovl = compute_mse(eta_true_grid_ovl.ravel(), eta_pred_flat_ovl)
    rate_ovl = compute_identifiable_rate(eta_pred_flat_ovl)
    
    print("\nResults:")
    print(f"  Scenario A (separated):")
    print(f"    MSE:               {mse_sep:.6f}")
    print(f"    Identifiable Rate: {rate_sep:.2%}")
    print(f"  Scenario B (overlap):")
    print(f"    MSE:               {mse_ovl:.6f}")
    print(f"    Identifiable Rate: {rate_ovl:.2%}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'scenario_A': {
            'mse': mse_sep,
            'identifiable_rate': rate_sep
        },
        'scenario_B': {
            'mse': mse_ovl,
            'identifiable_rate': rate_ovl
        }
    }
    with open('results/exp4_boundary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/exp4_boundary.json")
    
    # Visualizations
    os.makedirs('reports/figures', exist_ok=True)
    print("Generating figures...")
    
    plot_eta_2d(X_grid, eta_true_grid_sep, eta_pred_grid_sep, r_hat_grid_sep,
                save_path='reports/figures/exp4_separated.pdf')
    
    plot_eta_2d(X_grid, eta_true_grid_ovl, eta_pred_grid_ovl, r_hat_grid_ovl,
                save_path='reports/figures/exp4_overlap.pdf')
    
    print("\n" + "=" * 60)
    print("Experiment 4 completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()