"""
tests/test_compare_noise_types.py
Visual comparison using automatic Scott's Rule bandwidths.
"""
import numpy as np
import matplotlib.pyplot as plt
from src.data_generation import generate_data, eta_function
from src.estimation_em import EMNoiseEstimator
from src.bandwidth_selection import BandwidthSelector

def run_comparison():
    print("Running comparison with Scott's Rule Bandwidths...")
    
    n = 1500
    
    plt.figure(figsize=(14, 5))
    
    # --- CAS 1 : LINEAR ---
    plt.subplot(1, 2, 1)
    X, _, Z, _ = generate_data(n, d=1, noise_type='linear', noise_scale=1.0)
    
    h_s, h_n = BandwidthSelector.get_dual_bandwidths(X)
    
    model = EMNoiseEstimator(h_signal=h_s, h_noise=h_n, max_iter=30)
    model.fit(X, Z)
    
    X_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    eta_pred = model.predict_eta(X_grid)
    eta_true = eta_function(X_grid, noise_type='linear')
    
    plt.plot(X_grid, eta_true, 'g-', lw=3, label='True Noise')
    plt.plot(X_grid, eta_pred, 'r--', lw=2.5, label='EM Estimate')
    plt.title(f"Bruit Linéaire\n($h_s={h_s:.2f}, h_n={h_n:.2f}$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- CAS 2 : SINE ---
    plt.subplot(1, 2, 2)
    X, _, Z, _ = generate_data(n, d=1, noise_type='sine', noise_scale=1.0)
    
    h_s, h_n = BandwidthSelector.get_dual_bandwidths(X)
    
    model = EMNoiseEstimator(h_signal=h_s, h_noise=h_n, max_iter=30)
    model.fit(X, Z)
    
    eta_pred = model.predict_eta(X_grid)
    eta_true = eta_function(X_grid, noise_type='sine')
    
    plt.plot(X_grid, eta_true, 'g-', lw=3, label='True Noise')
    plt.plot(X_grid, eta_pred, 'r--', lw=2.5, label='EM Estimate')
    plt.title(f"Bruit Sinusoïdal\n($h_s={h_s:.2f}, h_n={h_n:.2f}$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tests/comparison_scott_rule.png')
    print("Graphique sauvegardé : tests/comparison_scott_rule.png")
    plt.show()

if __name__ == "__main__":
    run_comparison()