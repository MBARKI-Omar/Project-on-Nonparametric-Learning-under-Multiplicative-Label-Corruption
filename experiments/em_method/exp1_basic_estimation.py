"""
exp1_basic_estimation.py

Experiment 1: Basic Noise Function Estimation (d=1)

Objective:
    Demonstrate that the estimator works on a simple case and produces
    reasonable estimates of the noise function.

Validates:
    - The estimator is correctly implemented
    - eta_hat follows the shape of eta_true
    - Visual sanity check

Setup:
    - Sample size: m = 1000
    - Dimension: d = 1
    - Noise type: linear (varies from ~0 to ~0.4)
"""

import sys
import os

# Add parent directory to path for imports
# Modifie cette ligne :
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from src.config import ExperimentConfig
from src.utils import (
    setup_experiment, 
    generate_experiment_data,
    estimate_with_config,
    evaluate_with_config,
    generate_test_grid_1d,
    print_experiment_summary,
    save_results_json
)
from src.visualization import plot_eta_1d


def run_experiment():
    """
    Run Experiment 1: Basic estimation in 1D.
    """
    # ========================================================================
    # 1. SETUP
    # ========================================================================
    config = ExperimentConfig()
    setup_experiment(config)
    
    print("\n" + "="*70)
    print("Experiment 1: Basic Noise Function Estimation".center(70))
    print("="*70)
    
    # ========================================================================
    # 2. GENERATE DATA
    # ========================================================================
    print("\nGenerating data...")
    
    m = config.exp1_sample_size
    d = config.exp1_dimension
    noise_type = config.exp1_noise_type
    
    data = generate_experiment_data(
        m=m, 
        d=d, 
        noise_type=noise_type,
        seed=config.random_seed
    )
    
    print(f"  Sample size: {m}")
    print(f"  Dimension: {d}")
    print(f"  Noise type: {noise_type}")
    print(f"  Observed corruption rate: {np.mean(data['Y'] != data['Z']):.2%}")
    
    # ========================================================================
    # 3. GENERATE TEST GRID
    # ========================================================================
    X_test = generate_test_grid_1d(data['X'], n_points=config.exp1_n_test_points)
    x_test_1d = X_test[:, 0]  # For plotting
    
    # ========================================================================
    # 4. ESTIMATE NOISE FUNCTION
    # ========================================================================
    print("\nEstimating noise function...")
    
    eta_hat = estimate_with_config(
        X_test=X_test,
        X_train=data['X'],
        Z_train=data['Z'],
        config=config
    )
    
    print(f"  Bandwidth method: {config.bandwidth_method}")
    print(f"  Estimated eta_hat range: [{eta_hat.min():.3f}, {eta_hat.max():.3f}]")
    
    # ========================================================================
    # 5. EVALUATE
    # ========================================================================
    print("\nEvaluating performance...")
    
    metrics = evaluate_with_config(
        data_dict=data,
        eta_hat=eta_hat,
        X_test=X_test,
        noise_type=noise_type
    )
    
    # ========================================================================
    # 6. VISUALIZE
    # ========================================================================
    print("\nGenerating visualization...")
    
    # Get true eta at test points
    from src.data_generation import eta_function
    eta_true_test = eta_function(X_test, noise_type=noise_type)
    
    plot_eta_1d(
        x=x_test_1d,
        eta_true=eta_true_test,
        eta_hat=eta_hat,
        config=config,
        title=f"Experiment 1: Noise Function Estimation (m={m}, d={d})",
        filename="exp1_eta_estimation"
    )
    
    # ========================================================================
    # 7. SAVE RESULTS
    # ========================================================================
    results = {
        'experiment': 'exp1_basic_estimation',
        'config': {
            'sample_size': m,
            'dimension': d,
            'noise_type': noise_type,
            'bandwidth_method': config.bandwidth_method
        },
        'metrics': metrics,
        'observed_corruption_rate': float(np.mean(data['Y'] != data['Z']))
    }
    
    save_results_json(results, 'exp1_results', config)
    
    # ========================================================================
    # 8. PRINT SUMMARY
    # ========================================================================
    config_info = {
        'Sample size (m)': m,
        'Dimension (d)': d,
        'Noise type': noise_type,
        'Bandwidth method': config.bandwidth_method,
        'Corruption rate': f"{np.mean(data['Y'] != data['Z']):.2%}"
    }
    
    print_experiment_summary(
        title="Experiment 1: Results",
        config_info=config_info,
        results=metrics
    )
    
    print("✓ Experiment 1 completed successfully!\n")
    
    return results


if __name__ == "__main__":
    results = run_experiment()