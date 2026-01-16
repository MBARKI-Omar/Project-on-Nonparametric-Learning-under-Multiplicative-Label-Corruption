"""
Experiment 4: Bandwidth Selection Comparison

Compares different bandwidth selection strategies:
- Silverman's rule of thumb
- Cross-validation (LOOCV)
- Grid search with different h values
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


def silverman_bandwidth(X):
    """Silverman's rule of thumb."""
    m, d = X.shape
    std = np.std(X, axis=0).mean()
    h = ((4 / (d + 2)) ** (1 / (d + 4))) * std * (m ** (-1 / (d + 4)))
    return h


def run_experiment():
    """Compare bandwidth selection methods."""
    print("=" * 70)
    print("Experiment 4: Bandwidth Selection Comparison")
    print("=" * 70)
    
    # Configuration
    m, d = 1000, 2
    noise_type = 'linear'
    h_grid = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5])
    
    print(f"\nConfiguration:")
    print(f"  Sample size: {m}")
    print(f"  Dimension: {d}")
    print(f"  Bandwidth grid: {h_grid}")
    
    # Generate data
    np.random.seed(42)
    X, Y, Z, eta_true = generate_data(m, d=d, noise_type=noise_type)
    
    # Test all bandwidths
    results = {
        'h_values': h_grid.tolist(),
        'mse': [],
        'mae': [],
        'identifiable_rate': []
    }
    
    print("\n[Testing bandwidths...]")
    for h in h_grid:
        em = EMNoiseEstimator(
            h_signal=h,
            h_noise=h * 1.5,
            max_iter=30,
            tol=1e-4
        )
        em.fit(X, Z)
        eta_pred = em.predict_eta(X)
        
        mask = ~np.isnan(eta_pred)
        mse = np.mean((eta_pred[mask] - eta_true[mask])**2) if np.sum(mask) > 0 else np.nan
        mae = np.mean(np.abs(eta_pred[mask] - eta_true[mask])) if np.sum(mask) > 0 else np.nan
        id_rate = np.mean(mask)
        
        results['mse'].append(mse)
        results['mae'].append(mae)
        results['identifiable_rate'].append(id_rate)
        
        print(f"  h={h:.2f}: MSE={mse:.6f}, Identifiable={id_rate:.1%}")
    
    # Find optimal
    optimal_idx = np.nanargmin(results['mse'])
    h_optimal = h_grid[optimal_idx]
    mse_optimal = results['mse'][optimal_idx]
    
    # Silverman baseline
    h_silverman = silverman_bandwidth(X)
    em_silv = EMNoiseEstimator(h_signal=h_silverman, h_noise=h_silverman*1.5, max_iter=30)
    em_silv.fit(X, Z)
    eta_pred_silv = em_silv.predict_eta(X)
    mask_silv = ~np.isnan(eta_pred_silv)
    mse_silverman = np.mean((eta_pred_silv[mask_silv] - eta_true[mask_silv])**2)
    
    # Cross-validation
    print("\n[Running cross-validation...]")
    selector = BandwidthSelector(h_grid=h_grid)
    h_cv, cv_scores = selector.select_optimal_h(X, Z)
    
    em_cv = EMNoiseEstimator(h_signal=h_cv, h_noise=h_cv*1.5, max_iter=30)
    em_cv.fit(X, Z)
    eta_pred_cv = em_cv.predict_eta(X)
    mask_cv = ~np.isnan(eta_pred_cv)
    mse_cv = np.mean((eta_pred_cv[mask_cv] - eta_true[mask_cv])**2)
    
    # Save results
    results['h_silverman'] = h_silverman
    results['mse_silverman'] = mse_silverman
    results['h_cv'] = h_cv
    results['mse_cv'] = mse_cv
    results['h_optimal_grid'] = h_optimal
    results['mse_optimal_grid'] = mse_optimal
    
    os.makedirs('results/data', exist_ok=True)
    with open('results/data/exp4_bandwidth.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MSE vs h
    ax1.plot(h_grid, results['mse'], 'o-', linewidth=2, markersize=6, color='steelblue', label='MSE')
    ax1.axvline(h_optimal, color='red', linestyle='--', linewidth=2, label=f'Optimal h={h_optimal:.2f}')
    ax1.axvline(h_silverman, color='green', linestyle=':', linewidth=2, label=f'Silverman h={h_silverman:.2f}')
    ax1.axvline(h_cv, color='orange', linestyle='-.', linewidth=2, label=f'CV h={h_cv:.2f}')
    ax1.set_xlabel('Bandwidth h', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('MSE vs Bandwidth', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Method comparison
    methods = ['Grid\nOptimal', 'Silverman', 'Cross-Val']
    mse_values = [mse_optimal, mse_silverman, mse_cv]
    colors = ['red', 'green', 'orange']
    
    bars = ax2.bar(methods, mse_values, color=colors, alpha=0.7)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('Bandwidth Selection Methods Comparison', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.6f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs('results/figures/em_method', exist_ok=True)
    plt.savefig('results/figures/em_method/exp4_bandwidth_selection.pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: results/figures/em_method/exp4_bandwidth_selection.pdf")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOptimal h (grid search): {h_optimal:.3f} → MSE = {mse_optimal:.6f}")
    print(f"Silverman h:             {h_silverman:.3f} → MSE = {mse_silverman:.6f}")
    print(f"Cross-validation h:      {h_cv:.3f} → MSE = {mse_cv:.6f}")
    print("\n→ Bandwidth selection significantly impacts performance")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()