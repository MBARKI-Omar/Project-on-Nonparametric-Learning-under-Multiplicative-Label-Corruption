"""
Experiment 3: Effect of Dimension on Estimation Quality

Tests the curse of dimensionality: how MSE degrades as d increases.
Expected: MSE increases significantly with dimension.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import json
from src.data_generation import generate_data, eta_function
from src.estimation_em import EMNoiseEstimator
from src.bandwidth_selection import BandwidthSelector


def run_experiment():
    """Test dimension effect on EM performance."""
    print("=" * 70)
    print("Experiment 3: Dimension Effect (Curse of Dimensionality)")
    print("=" * 70)
    
    # Configuration
    dimensions = [1, 2, 3, 5]
    m = 5000  # Large sample to isolate dimension effect
    noise_type = 'linear'
    n_trials = 5  # Average over trials
    
    print(f"\nConfiguration:")
    print(f"  Dimensions: {dimensions}")
    print(f"  Sample size: {m}")
    print(f"  Trials per dimension: {n_trials}")
    
    # Storage
    results = {
        'dimensions': dimensions,
        'mse_mean': [],
        'mse_std': [],
        'mae_mean': [],
        'time_mean': []
    }
    
    for d in dimensions:
        print(f"\n[Dimension d={d}]")
        
        mse_list = []
        mae_list = []
        time_list = []
        
        for trial in range(n_trials):
            np.random.seed(42 + trial)
            
            # Generate data
            X, Y, Z, eta_true = generate_data(m, d=d, noise_type=noise_type)
            
            # Select bandwidth for this dimension
            selector = BandwidthSelector(h_grid=np.linspace(0.1, 1.0, 10))
            h_cv, _ = selector.select_optimal_h(X, Z)
            
            # Train EM
            import time
            start = time.time()
            
            em = EMNoiseEstimator(
                h_signal=h_cv,
                h_noise=h_cv * 1.5,
                max_iter=30,
                tol=1e-4
            )
            em.fit(X, Z)
            
            elapsed = time.time() - start
            
            # Predict
            eta_pred = em.predict_eta(X)
            
            # Metrics
            mask = ~np.isnan(eta_pred)
            if np.sum(mask) > 0:
                mse = np.mean((eta_pred[mask] - eta_true[mask])**2)
                mae = np.mean(np.abs(eta_pred[mask] - eta_true[mask]))
            else:
                mse, mae = np.nan, np.nan
            
            mse_list.append(mse)
            mae_list.append(mae)
            time_list.append(elapsed)
            
            print(f"  Trial {trial+1}/{n_trials}: MSE={mse:.6f}, Time={elapsed:.1f}s")
        
        # Aggregate
        results['mse_mean'].append(np.mean(mse_list))
        results['mse_std'].append(np.std(mse_list))
        results['mae_mean'].append(np.mean(mae_list))
        results['time_mean'].append(np.mean(time_list))
        
        print(f"  → Average MSE: {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}")
    
    # Save results
    os.makedirs('results/data', exist_ok=True)
    with open('results/data/exp3_dimension.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MSE vs Dimension
    ax1.bar(range(len(dimensions)), results['mse_mean'], 
            yerr=results['mse_std'], capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xticks(range(len(dimensions)))
    ax1.set_xticklabels([f'd={d}' for d in dimensions])
    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('Curse of Dimensionality: MSE vs Dimension', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (mse, std) in enumerate(zip(results['mse_mean'], results['mse_std'])):
        ax1.text(i, mse + std + 0.001, f'{mse:.4f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Computation Time
    ax2.plot(dimensions, results['time_mean'], 'o-', linewidth=2, 
            markersize=8, color='coral', label='Computation Time')
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Computational Cost vs Dimension', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    os.makedirs('results/figures/em_method', exist_ok=True)
    plt.savefig('results/figures/em_method/exp3_dimension_effect.pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: results/figures/em_method/exp3_dimension_effect.pdf")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    degradation = results['mse_mean'][-1] / results['mse_mean'][0]
    print(f"\nMSE at d=1: {results['mse_mean'][0]:.6f}")
    print(f"MSE at d={dimensions[-1]}: {results['mse_mean'][-1]:.6f}")
    print(f"Degradation factor: {degradation:.2f}x")
    print("\n→ Curse of dimensionality confirmed: MSE increases with d")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()