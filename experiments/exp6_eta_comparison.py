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
    """Compare true eta vs estimated eta directly."""
    print("=" * 60)
    print("Experiment 6: True vs Estimated Eta Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data over MULTIPLE periods
    m, d = 2000, 1  # Use 1D for clear sinusoid visualization
    noise_level = 'medium'
    
    print(f"\nGenerating data: m={m}, d={d}, noise_level={noise_level}")
    
    # Generate X over 3 periods: [0, 3] instead of [0, 1]
    X = np.random.uniform(0, 3, size=(m, d))
    
    # Generate Y (using standard f_function on normalized x)
    from src.data_generation import f_function, generate_Z
    X_normalized = X / 3.0  # Normalize to [0,1] for f_function
    probs = f_function(X_normalized)
    Y = np.where(np.random.rand(m) < probs, 1, -1)
    
    # Generate Z with eta over 3 periods
    def eta_3periods(x):
        """Eta function over 3 periods."""
        return 0.20 + 0.15 * np.sin(np.pi * x.ravel())  # One full period in [0,3]
    
    noise_probs = eta_3periods(X)
    flip = np.random.rand(m) < noise_probs
    Z = np.where(flip, -Y, Y)
    
    # Create test grid over 3 periods
    X_test = np.linspace(0, 3, 300).reshape(-1, 1)
    
    # Compute true eta on test points
    eta_true_test = eta_3periods(X_test)
    
    # Estimate eta
    print("Estimating eta...")
    eta_pred_test = estimate_eta(X, Z, X_test, h=None, threshold=0.3)
    
    # Statistics
    mask = ~np.isnan(eta_pred_test)
    if np.sum(mask) > 0:
        mse = np.mean((eta_true_test[mask] - eta_pred_test[mask])**2)
        mae = np.mean(np.abs(eta_true_test[mask] - eta_pred_test[mask]))
        print(f"\nMetrics on test points:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Identifiable: {np.mean(mask):.1%}")
    
    # Print actual values at a few points
    print("\nSample comparisons (evenly spaced points):")
    print("  x     | True η | Estimated η | Error")
    indices = np.linspace(0, len(X_test)-1, 10, dtype=int)
    for i in indices:
        if not np.isnan(eta_pred_test[i]):
            error = eta_pred_test[i] - eta_true_test[i]
            print(f"  {X_test[i,0]:.2f} | {eta_true_test[i]:.4f} | {eta_pred_test[i]:.4f}   | {error:+.4f}")
        else:
            print(f"  {X_test[i,0]:.2f} | {eta_true_test[i]:.4f} | NaN       | NaN")
    
    # Visualization
    os.makedirs('reports/figures', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_axis = X_test[:, 0]
    
    # Plot true function (smooth sinusoid over 3 periods)
    ax.plot(x_axis, eta_true_test, 'b-', linewidth=3, label='True η(x) = 0.20 + 0.15·sin(πx)', alpha=0.8)
    
    # Plot estimated values (only non-NaN)
    mask = ~np.isnan(eta_pred_test)
    ax.plot(x_axis[mask], eta_pred_test[mask], 'go', markersize=4, 
            label='Estimated η̂(x)', alpha=0.6)
    
    # Mark non-identifiable points
    if np.sum(~mask) > 0:
        ax.plot(x_axis[~mask], eta_true_test[~mask], 'rx', markersize=6,
                label='Non-identifiable', alpha=0.5)
    
    # Show training data distribution (rug plot at bottom)
    ax.scatter(X[:, 0], np.zeros(m) + 0.03, c='lightgray', s=1, alpha=0.3, 
              label='Training points', marker='|')
    
    # Mark periods
    for period in [1, 2]:
        ax.axvline(period, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('x (over 3 periods)', fontsize=12)
    ax.set_ylabel('η(x)', fontsize=12)
    ax.set_title('Sinusoidal Noise Function: True vs Estimated (3 periods)', fontsize=14)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 0.4)
    ax.legend(fontsize=11)
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
    if np.sum(mask) > 0 and mse < 0.001:
        print("✅ Estimator tracks sinusoid well (MSE < 0.001)")
    elif np.sum(mask) > 0 and mse < 0.01:
        print("⚠️  Estimator has moderate error (0.001 < MSE < 0.01)")
    elif np.sum(mask) > 0:
        print(f"❌ Estimator does NOT track sinusoid (MSE = {mse:.4f})")
    else:
        print("❌ No identifiable points - threshold too high or data issue")

if __name__ == "__main__":
    run_experiment()