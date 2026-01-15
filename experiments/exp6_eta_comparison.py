"""
Experiment 6: Direct comparison of true vs estimated eta.
Visual diagnostic to check if estimator tracks the true noise function.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation import generate_data, eta_function
from src.estimator import estimate_eta


def run_experiment():
    """Compare true eta vs estimated eta directly over multiple periods."""
    print("=" * 60)
    print("Experiment 6: True vs Estimated Eta Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data in standard [0, 1] range
    m, d = 2000, 1
    noise_level = 'medium'
    
    print(f"\nGenerating data: m={m}, d={d}, noise_level={noise_level}")
    X, Y, Z = generate_data(m, d, noise_level=noise_level)
    
    # Create test grid: replicate [0,1] pattern 3 times for visualization
    n_periods = 3
    n_points_per_period = 100
    X_test_single = np.linspace(0, 1, n_points_per_period).reshape(-1, 1)
    
    # Replicate pattern 3 times
    X_test_list = []
    x_axis_list = []
    
    for period in range(n_periods):
        X_test_list.append(X_test_single)
        x_axis_list.append(X_test_single[:, 0] + period)
    
    X_test = np.vstack(X_test_list)
    x_axis = np.concatenate(x_axis_list)
    
    # Compute true eta (same for all periods since pattern repeats)
    eta_true_single = eta_function(X_test_single, noise_level=noise_level)
    eta_true_test = np.tile(eta_true_single, n_periods)
    
    # Estimate eta (same pattern repeated)
    print("Estimating eta...")
    eta_pred_single = estimate_eta(X, Z, X_test_single, h=None, threshold=0.3)
    eta_pred_test = np.tile(eta_pred_single, n_periods)
    
    # Statistics on identifiable points
    mask = ~np.isnan(eta_pred_test)
    if np.sum(mask) > 0:
        mse = np.mean((eta_true_test[mask] - eta_pred_test[mask])**2)
        mae = np.mean(np.abs(eta_true_test[mask] - eta_pred_test[mask]))
        print(f"\nMetrics on test points:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Identifiable: {np.mean(mask):.1%}")
        
        # Check if estimator tracks well
        if mse < 0.001:
            quality = "EXCELLENT"
        elif mse < 0.005:
            quality = "GOOD"
        elif mse < 0.01:
            quality = "MODERATE"
        else:
            quality = "POOR"
        print(f"  Quality: {quality}")
    else:
        print("\n⚠️  No identifiable points!")
        mse = np.nan
        quality = "FAILED"
    
    # Print sample values
    print("\nSample comparisons (one per period):")
    print("  Period | x    | True η | Est η  | Error")
    print("  " + "-" * 50)
    sample_indices = [25, 125, 225]  # Middle of each period
    for idx in sample_indices:
        period = idx // n_points_per_period
        x_val = x_axis[idx]
        eta_true_val = eta_true_test[idx]
        eta_pred_val = eta_pred_test[idx]
        
        if not np.isnan(eta_pred_val):
            error = eta_pred_val - eta_true_val
            print(f"  {period}      | {x_val:.2f} | {eta_true_val:.4f} | {eta_pred_val:.4f} | {error:+.4f}")
        else:
            print(f"  {period}      | {x_val:.2f} | {eta_true_val:.4f} | NaN    | NaN")
    
    # Visualization
    os.makedirs('reports/figures', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot true function (smooth sinusoid)
    ax.plot(x_axis, eta_true_test, 'b-', linewidth=3, 
            label='True η(x) = 0.20 + 0.15·sin(πx)', alpha=0.8)
    
    # Plot estimated values (only non-NaN)
    mask = ~np.isnan(eta_pred_test)
    ax.plot(x_axis[mask], eta_pred_test[mask], 'go', markersize=4, 
            label='Estimated η̂(x)', alpha=0.6)
    
    # Mark non-identifiable points
    if np.sum(~mask) > 0:
        ax.plot(x_axis[~mask], eta_true_test[~mask], 'rx', markersize=6,
                label='Non-identifiable', alpha=0.5)
    
    # Show training data distribution (rug plot)
    X_train_extended = X[:, 0]
    for period in range(n_periods):
        ax.scatter(X_train_extended + period, np.zeros(len(X_train_extended)) + 0.03, 
                  c='lightgray', s=1, alpha=0.2, marker='|')
    
    # Mark period boundaries
    for period in [1, 2]:
        ax.axvline(period, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add text annotation with quality
    ax.text(0.02, 0.98, f'MSE = {mse:.6f}\nQuality: {quality}', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('x (3 periods of the pattern)', fontsize=12)
    ax.set_ylabel('η(x)', fontsize=12)
    ax.set_title('Sinusoidal Noise: True vs Estimated (Pattern repeated 3 times)', fontsize=14)
    ax.set_xlim(0, n_periods)
    ax.set_ylim(0, 0.4)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = 'reports/figures/exp6_eta_comparison.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Experiment 6 completed!")
    print("=" * 60)
    print("\nDIAGNOSTIC:")
    if quality == "EXCELLENT":
        print("✅ Estimator tracks the sinusoid PERFECTLY")
        print("   → Your method works well!")
    elif quality == "GOOD":
        print("✅ Estimator tracks the sinusoid WELL")
        print("   → Your method is solid")
    elif quality == "MODERATE":
        print("⚠️  Estimator has moderate tracking error")
        print("   → Method works but has room for improvement")
    elif quality == "POOR":
        print("❌ Estimator does NOT track the sinusoid well")
        print("   → There may be an implementation issue")
    else:
        print("❌ Estimator failed completely")
        print("   → Check threshold or implementation")


if __name__ == "__main__":
    run_experiment()