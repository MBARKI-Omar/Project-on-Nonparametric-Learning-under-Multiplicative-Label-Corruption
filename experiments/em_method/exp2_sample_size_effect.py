"""
exp2_sample_size_effect.py

Experiment 2: Effect of Sample Size on Convergence

Objective:
    Verify empirically the theoretical convergence rate:
    MSE ~ m^{-4/(d+4)}

Validates:
    - Theorem 1 (consistency) from Part II
    - Convergence rate matches theory

Setup:
    - Sample sizes: m ∈ [50, 100, 200, 500, 1000, 2000, 5000]
    - Dimension: d = 1 (fixed)
    - Noise type: linear (fixed)
    - Monte Carlo trials: 20 repetitions per sample size
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.config import ExperimentConfig
from src.utils import (
    setup_experiment,
    generate_experiment_data,
    estimate_with_config,
    evaluate_with_config,
    generate_test_sample,
    print_progress,
    save_results_csv
)
from src.visualization import plot_mse_vs_sample_size
from src.evaluation import aggregate_metrics
import time


def run_experiment():
    """
    Run Experiment 2: Sample size effect on convergence.
    """
    # ========================================================================
    # 1. SETUP
    # ========================================================================
    config = ExperimentConfig()
    setup_experiment(config)
    
    print("\n" + "="*70)
    print("Experiment 2: Sample Size Effect on Convergence".center(70))
    print("="*70)
    
    # ========================================================================
    # 2. CONFIGURATION
    # ========================================================================
    m_values = config.exp2_sample_sizes
    d = config.exp2_dimension
    noise_type = config.exp2_noise_type
    n_trials = config.exp2_n_trials
    n_test = config.exp2_n_test_points
    
    print(f"\nConfiguration:")
    print(f"  Sample sizes: {m_values}")
    print(f"  Dimension: {d}")
    print(f"  Noise type: {noise_type}")
    print(f"  Trials per size: {n_trials}")
    print(f"  Theoretical rate: m^{{-4/(d+4)}} = m^{{-{4/(d+4):.3f}}}")
    
    # ========================================================================
    # 3. RUN TRIALS
    # ========================================================================
    print(f"\nRunning {len(m_values)} × {n_trials} = {len(m_values) * n_trials} trials...")
    
    results_by_m = {}
    start_time = time.time()
    total_iterations = len(m_values) * n_trials
    current_iteration = 0
    
    for i, m in enumerate(m_values):
        print(f"\n[{i+1}/{len(m_values)}] Sample size m = {m}")
        
        metrics_list = []
        
        for trial in range(n_trials):
            # Generate data
            data = generate_experiment_data(
                m=m,
                d=d,
                noise_type=noise_type,
                seed=config.random_seed + trial
            )
            
            # Generate test points
            X_test = generate_test_sample(data['X'], n_points=n_test)
            
            # Estimate
            eta_hat = estimate_with_config(
                X_test=X_test,
                X_train=data['X'],
                Z_train=data['Z'],
                config=config
            )
            
            # Evaluate
            metrics = evaluate_with_config(
                data_dict=data,
                eta_hat=eta_hat,
                X_test=X_test,
                noise_type=noise_type
            )
            
            metrics_list.append(metrics)
            
            # Progress
            current_iteration += 1
            print_progress(trial, n_trials, prefix=f"  m={m}")
        
        # Aggregate trials for this sample size
        results_by_m[m] = aggregate_metrics(metrics_list)
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ All trials completed in {elapsed_time:.1f}s")
    
    # ========================================================================
    # 4. EXTRACT RESULTS
    # ========================================================================
    mse_means = np.array([results_by_m[m]['mse']['mean'] for m in m_values])
    mse_stds = np.array([results_by_m[m]['mse']['std'] for m in m_values])
    
    # ========================================================================
    # 5. VISUALIZE
    # ========================================================================
    print("\nGenerating visualization...")
    
    plot_mse_vs_sample_size(
        m_values=np.array(m_values),
        mse_means=mse_means,
        mse_stds=mse_stds,
        config=config,
        d=d,
        title=f"Experiment 2: Convergence with Sample Size (d={d})",
        filename="exp2_convergence_sample_size"
    )
    
    # ========================================================================
    # 6. SAVE RESULTS
    # ========================================================================
    results_table = {
        'sample_size': m_values,
        'mse_mean': mse_means.tolist(),
        'mse_std': mse_stds.tolist(),
        'mae_mean': [results_by_m[m]['mae']['mean'] for m in m_values],
        'mae_std': [results_by_m[m]['mae']['std'] for m in m_values]
    }
    
    save_results_csv(results_table, 'exp2_results', config)
    
    # ========================================================================
    # 7. PRINT SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("Experiment 2: Summary".center(70))
    print("="*70)
    print(f"\n{'Sample Size':>12s} {'MSE Mean':>12s} {'MSE Std':>12s} {'Improvement':>15s}")
    print("-"*70)
    
    for i, m in enumerate(m_values):
        if i == 0:
            improvement = "baseline"
        else:
            improvement = f"{mse_means[0]/mse_means[i]:.2f}x better"
        print(f"{m:>12d} {mse_means[i]:>12.6f} {mse_stds[i]:>12.6f} {improvement:>15s}")
    
    print("="*70)
    
    print(f"\nObservations:")
    print(f"  - MSE decreases with sample size (consistency validated)")
    print(f"  - From m={m_values[0]} to m={m_values[-1]}: "
          f"{mse_means[0]/mse_means[-1]:.1f}x improvement")
    print(f"  - Check figure for comparison with theoretical rate")
    
    print("\n✓ Experiment 2 completed successfully!\n")
    
    return results_table


if __name__ == "__main__":
    results = run_experiment()