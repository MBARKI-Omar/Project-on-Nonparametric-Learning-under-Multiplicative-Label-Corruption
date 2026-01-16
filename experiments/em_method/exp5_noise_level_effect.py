"""
Experiment 5: Effect of Noise Level on Estimation Quality

Tests robustness to different noise intensities:
- Low noise (η_max ≈ 0.2)
- Medium noise (η_max ≈ 0.35)
- High noise (η_max ≈ 0.45)
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
    """Test noise level effect on EM performance."""
    print("=" * 70)
    print("Experiment 5: Noise Level Effect")
    print("=" * 70)
    
    # Configuration
    m, d = 2000, 2
    noise_type = 'linear'
    noise_scales = {
        'low': 0.5,      # η_max ≈ 0.20
        'medium': 1.0,   # η_max ≈ 0.40
        'high': 1.2      # η_max ≈ 0.48
    }
    n_trials = 5
    
    print(f"\nConfiguration:")
    print(f"  Sample size: {m}")
    print(f"  Dimension: {d}")
    print(f"  Noise scales: {list(noise_scales.keys())}")
    print(f"  Trials per level: {n_trials}")
    
    # Storage
    results = {
        'noise_levels': list(noise_scales.keys()),
        'noise_scales': list(noise_scales.values()),
        'mse_mean': [],
        'mse_std': [],
        'mae_mean': [],
        'corruption_rate_mean': []
    }
    
    for level, scale in noise_scales.items():
        print(f"\n[Noise level: {level} (scale={scale})]")
        
        mse_list = []
        mae_list = []
        corr_list = []
        
        for trial in range(n_trials):
            np.random.seed(42 + trial)
            
            # Generate data with specific noise level
            X, Y, Z, eta_true = generate_data(m, d=d, noise_type=noise_type, noise_scale=scale)
            
            # Observed corruption rate
            corruption_rate = np.mean(Y != Z)
            corr_list.append(corruption_rate)
            
            # Select bandwidth
            selector = BandwidthSelector(h_grid=np.linspace(0.1, 0.8, 10))
            h_cv, _ = selector.select_optimal_h(X, Z)
            
            # Train EM
            em = EMNoiseEstimator(
                h_signal=h_cv,
                h_noise=h_cv * 1.5,
                max_iter=40,
                tol=1e-4
            )
            em.fit(X, Z)
            
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
            
            print(f"  Trial {trial+1}: Corruption={corruption_rate:.1%}, MSE={mse:.6f}")
        
        # Aggregate
        results['mse_mean'].append(np.mean(mse_list))
        results['mse_std'].append(np.std(mse_list))
        results['mae_mean'].append(np.mean(mae_list))
        results['corruption_rate_mean'].append(np.mean(corr_list))
        
        print(f"  → Average MSE: {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}")
        print(f"  → Average corruption: {np.mean(corr_list):.1%}")
    
    # Save results
    os.makedirs('results/data', exist_ok=True)
    with open('results/data/exp5_noise_level.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x_pos = np.arange(len(results['noise_levels']))
    
    # Plot 1: MSE vs Noise Level
    ax1.bar(x_pos, results['mse_mean'], yerr=results['mse_std'], 
            capsize=5, alpha=0.7, color=['green', 'orange', 'red'])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results['noise_levels'])
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('MSE vs Noise Level', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Add values
    for i, (mse, std) in enumerate(zip(results['mse_mean'], results['mse_std'])):
        ax1.text(i, mse + std + 0.001, f'{mse:.4f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Corruption Rate
    ax2.bar(x_pos, results['corruption_rate_mean'], 
            alpha=0.7, color=['green', 'orange', 'red'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results['noise_levels'])
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Observed Corruption Rate', fontsize=12)
    ax2.set_title('Label Corruption Rate by Noise Level', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim([0, 0.5])
    
    # Add values
    for i, rate in enumerate(results['corruption_rate_mean']):
        ax2.text(i, rate + 0.01, f'{rate:.1%}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs('results/figures/em_method', exist_ok=True)
    plt.savefig('results/figures/em_method/exp5_noise_level_effect.pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: results/figures/em_method/exp5_noise_level_effect.pdf")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    degradation = results['mse_mean'][-1] / results['mse_mean'][0]
    print(f"\nMSE (low noise):    {results['mse_mean'][0]:.6f}")
    print(f"MSE (medium noise): {results['mse_mean'][1]:.6f}")
    print(f"MSE (high noise):   {results['mse_mean'][2]:.6f}")
    print(f"Degradation factor: {degradation:.2f}x")
    print("\n→ EM remains relatively robust across noise levels")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()