"""
Experiment 5: Effect of noise level on estimation quality.
"""

import numpy as np
import json

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation import generate_data, eta_function
from src.evaluation import evaluate_estimator
from src.visualization import plot_noise_level_effect


def run_experiment():
    """Run noise level effect experiment."""
    print("=" * 60)
    print("Experiment 5: Noise Level Effect")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test different noise levels
    noise_levels = ['low', 'medium', 'high']
    m, d = 2000, 2
    
    print(f"\nTesting different noise levels (m={m}, d={d})...")
    metrics_list = []
    
    for level in noise_levels:
        print(f"  Noise level: {level}...", end='')
        X, Y, Z = generate_data(m, d, noise_level=level)
        
        # Use corresponding eta function for evaluation
        eta_func = lambda x: eta_function(x, noise_level=level)
        metrics = evaluate_estimator(X, Z, X, eta_func, h=None, threshold=0.3)
        metrics_list.append(metrics)
        
        # Also report empirical corruption rate
        corruption_rate = np.mean(Y != Z)
        print(f" MSE = {metrics['mse']:.6f}, Corruption = {corruption_rate:.2%}")
    
    # Show degradation
    print("\nMSE increase with noise level:")
    for i, level in enumerate(noise_levels):
        if i == 0:
            print(f"  {level}: MSE = {metrics_list[i]['mse']:.6f} (baseline)")
        else:
            ratio = metrics_list[i]['mse'] / metrics_list[0]['mse']
            print(f"  {level}: MSE = {metrics_list[i]['mse']:.6f} ({ratio:.2f}x worse)")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'noise_levels': noise_levels,
        'metrics': metrics_list
    }
    with open('results/exp5_noise_level.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/exp5_noise_level.json")
    
    # Visualization
    os.makedirs('reports/figures', exist_ok=True)
    print("Generating figure...")
    plot_noise_level_effect(noise_levels, metrics_list,
                           save_path='reports/figures/exp5_noise_level_effect.pdf')
    
    print("\n" + "=" * 60)
    print("Experiment 5 completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()