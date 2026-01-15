"""
Experiment 1: Effect of bandwidth h on estimation quality.
"""

import numpy as np
import json

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation import generate_data, eta_function
from src.evaluation import evaluate_estimator
from src.visualization import plot_bandwidth_effect


def run_experiment():
    """Run bandwidth selection experiment."""
    print("=" * 60)
    print("Experiment 1: Bandwidth Selection")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate fixed dataset
    m, d = 1000, 2
    print(f"\nGenerating data: m={m}, d={d}")
    X, Y, Z = generate_data(m, d, noise_level='medium')
    
    # Test different bandwidths
    h_values = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    
    print("\nTesting different bandwidths...")
    metrics_list = []
    
    for h in h_values:
        print(f"  h = {h:.2f}...", end='')
        metrics = evaluate_estimator(X, Z, X, eta_function, h=h, threshold=0.3)
        metrics_list.append(metrics)
        print(f" MSE = {metrics['mse']:.6f}")
    
    # Find optimal
    mse_values = [m['mse'] for m in metrics_list]
    opt_idx = np.argmin(mse_values)
    print(f"\nOptimal bandwidth: h = {h_values[opt_idx]:.2f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'h_values': h_values,
        'metrics': metrics_list
    }
    with open('results/exp1_bandwidth.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/exp1_bandwidth.json")
    
    # Visualization
    os.makedirs('reports/figures', exist_ok=True)
    print("\nGenerating figure...")
    plot_bandwidth_effect(h_values, metrics_list, 
                         save_path='reports/figures/exp1_bandwidth_effect.pdf')
    
    print("\n" + "=" * 60)
    print("Experiment 1 completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()