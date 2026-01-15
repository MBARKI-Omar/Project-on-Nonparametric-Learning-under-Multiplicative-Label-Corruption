"""
Experiment 3: Curse of dimensionality - effect of dimension d.
"""

import numpy as np
import json

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation import generate_data, eta_function
from src.evaluation import evaluate_estimator
from src.visualization import plot_dimension_effect


def run_experiment():
    """Run dimension effect experiment."""
    print("=" * 60)
    print("Experiment 3: Dimension Effect (Curse of Dimensionality)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test different dimensions
    d_values = [1, 2, 3, 5]
    m = 5000  # Large sample to see pure dimension effect
    
    print(f"\nTesting different dimensions (m={m})...")
    metrics_list = []
    
    for d in d_values:
        print(f"  d = {d}...", end='')
        X, Y, Z = generate_data(m, d, noise_level='medium')
        metrics = evaluate_estimator(X, Z, X, eta_function, h=None, threshold=0.3)
        metrics_list.append(metrics)
        print(f" MSE = {metrics['mse']:.6f}")
    
    # Show degradation
    print("\nMSE increase with dimension:")
    for i, d in enumerate(d_values):
        if i == 0:
            print(f"  d={d}: MSE = {metrics_list[i]['mse']:.6f} (baseline)")
        else:
            ratio = metrics_list[i]['mse'] / metrics_list[0]['mse']
            print(f"  d={d}: MSE = {metrics_list[i]['mse']:.6f} ({ratio:.2f}x worse)")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'd_values': d_values,
        'metrics': metrics_list,
        'm': m
    }
    with open('results/exp3_dimension.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/exp3_dimension.json")
    
    # Visualization
    os.makedirs('reports/figures', exist_ok=True)
    print("Generating figure...")
    plot_dimension_effect(d_values, metrics_list,
                         save_path='reports/figures/exp3_curse_dimensionality.pdf')
    
    print("\n" + "=" * 60)
    print("Experiment 3 completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()