"""
Experiment 2: Effect of sample size m on convergence rate.
"""

import numpy as np
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation import generate_data, eta_function
from src.evaluation import evaluate_estimator
from src.visualization import plot_sample_size_effect


def run_experiment():
    """Run sample size convergence experiment."""
    print("=" * 60)
    print("Experiment 2: Sample Size Effect")
    print("=" * 60)
    
    # Test different sample sizes
    m_values = [100, 200, 500, 1000, 2000, 5000]
    d = 2
    n_repeats = 10  # Average over 10 runs to reduce variance
    
    print(f"\nTesting different sample sizes (d={d}, {n_repeats} repeats per m)...")
    metrics_list = []
    
    for m in m_values:
        print(f"  m = {m}...", end='')
        
        # Run multiple times and average
        mse_values = []
        for rep in range(n_repeats):
            seed = 42 + rep  # Different seed per repeat
            np.random.seed(seed)
            
            X, Y, Z = generate_data(m, d, noise_level='medium')
            metrics = evaluate_estimator(X, Z, X, eta_function, h=None, threshold=0.3)
            mse_values.append(metrics['mse'])
        
        # Average metrics
        avg_mse = np.mean(mse_values)
        avg_metrics = {
            'mse': avg_mse,
            'mae': metrics['mae'],  # Use last one (representative)
            'rmse': np.sqrt(avg_mse),
            'max_error': metrics['max_error'],
            'identifiable_rate': metrics['identifiable_rate']
        }
        metrics_list.append(avg_metrics)
        
        print(f" MSE = {avg_mse:.6f} (avg over {n_repeats} runs)")
    
    # Estimate convergence rate
    log_m = np.log(m_values)
    log_mse = np.log([m['mse'] for m in metrics_list])
    slope, _ = np.polyfit(log_m, log_mse, 1)
    
    theoretical_rate = -4 / (d + 4)  # -0.67 for d=2
    
    print(f"\nEmpirical convergence rate: {slope:.3f}")
    print(f"Theoretical rate:           {theoretical_rate:.3f}")
    print(f"Relative error:             {abs(slope - theoretical_rate) / abs(theoretical_rate):.1%}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'm_values': m_values,
        'metrics': metrics_list,
        'empirical_rate': slope,
        'theoretical_rate': theoretical_rate,
        'n_repeats': n_repeats
    }
    with open('results/exp2_sample_size.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/exp2_sample_size.json")
    
    # Visualization
    os.makedirs('reports/figures', exist_ok=True)
    print("Generating figure...")
    plot_sample_size_effect(m_values, metrics_list,
                           save_path='reports/figures/exp2_convergence_rate.pdf')
    
    print("\n" + "=" * 60)
    print("Experiment 2 completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()