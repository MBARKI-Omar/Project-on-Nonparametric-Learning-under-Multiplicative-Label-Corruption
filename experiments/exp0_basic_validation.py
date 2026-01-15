"""
Experiment 0: Basic validation of the estimator.
Demonstrates that the method works on a simple case.
"""

import numpy as np
import json

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation import generate_data, eta_function
from src.evaluation import evaluate_estimator
from src.visualization import plot_eta_1d, plot_eta_2d
from src.estimator import estimate_eta, nadaraya_watson


def run_experiment():
    """Run basic validation experiment."""
    print("=" * 60)
    print("Experiment 0: Basic Validation")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    m, d = 2000, 2
    print(f"\nGenerating data: m={m}, d={d}")
    X, Y, Z = generate_data(m, d, noise_level='medium')
    
    # Evaluate estimator
    print("Estimating noise function...")
    metrics = evaluate_estimator(X, Z, X, eta_function, threshold=0.3)
    
    print("\nResults:")
    print(f"  MSE:               {metrics['mse']:.6f}")
    print(f"  MAE:               {metrics['mae']:.6f}")
    print(f"  RMSE:              {metrics['rmse']:.6f}")
    print(f"  Max Error:         {metrics['max_error']:.6f}")
    print(f"  Identifiable Rate: {metrics['identifiable_rate']:.2%}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/exp0_basic_validation.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nResults saved to results/exp0_basic_validation.json")
    
    # Visualizations
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1D visualization (first dimension only)
    print("\nGenerating 1D visualization...")
    X_1d = X[:500, 0:1]  # First 500 points, first dimension
    eta_true_1d = eta_function(X_1d)
    eta_pred_1d = estimate_eta(X, Z, X_1d, threshold=0.3)
    plot_eta_1d(X_1d, eta_true_1d, eta_pred_1d, 
                save_path='reports/figures/exp0_validation_1d.pdf')
    
    # 2D visualization on grid
    print("Generating 2D visualization...")
    grid_size = 50
    x1 = np.linspace(0, 1, grid_size)
    x2 = np.linspace(0, 1, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    eta_true_grid = eta_function(X_grid).reshape(grid_size, grid_size)
    eta_pred_flat = estimate_eta(X, Z, X_grid, threshold=0.3)
    eta_pred_grid = eta_pred_flat.reshape(grid_size, grid_size)
    
    # Also compute r_hat for visualization
    r_hat_flat = nadaraya_watson(X, Z, X_grid, h=None)
    r_hat_grid = r_hat_flat.reshape(grid_size, grid_size)
    
    plot_eta_2d(X_grid, eta_true_grid, eta_pred_grid, r_hat_grid,
                save_path='reports/figures/exp0_validation_2d.pdf')
    
    print("\n" + "=" * 60)
    print("Experiment 0 completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()