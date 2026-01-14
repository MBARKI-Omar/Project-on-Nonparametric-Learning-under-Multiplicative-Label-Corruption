"""
tests/test_performance_analysis.py

Performance analysis using Cross-Validation at every step.
Evaluates MSE vs Sample Size, Dimension, and Noise Level.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.data_generation import generate_data
from src.estimation_em import EMNoiseEstimator
from src.bandwidth_selection import BandwidthSelector

def compute_mse(estimator, X_test, eta_true_test):
    eta_pred = estimator.predict_eta(X_test)
    return np.mean((eta_pred - eta_true_test)**2)

def run_performance_suite():
    print("==================================================")
    print("PERFORMANCE ANALYSIS (Full Cross-Validation)")
    print("==================================================")
    
    noise_type = 'sine'
    n_test = 1000
    
    # [1/3] Sample Size (m)
    print("\n[1/3] Influence of Sample Size (m)...")
    m_values = [100, 300, 500, 1000, 2000]
    mse_m = []
    
    X_test_m, _, _, eta_test_m = generate_data(n_test, d=1, noise_type=noise_type)
    
    for m in m_values:
        X_train, _, Z_train, _ = generate_data(m, d=1, noise_type=noise_type)
        
        # CV Step
        selector = BandwidthSelector(h_grid=np.linspace(0.1, 1.0, 15))
        h_cv, _ = selector.select_optimal_h(X_train, Z_train)
        
        model = EMNoiseEstimator(h_signal=h_cv, h_noise=h_cv * 1.5, max_iter=20)
        model.fit(X_train, Z_train)
        
        err = compute_mse(model, X_test_m, eta_test_m)
        mse_m.append(err)
        print(f"   m={m}: h_cv={h_cv:.3f} -> MSE={err:.5f}")

    # [2/3] Dimension (d)
    print("\n[2/3] Influence of Dimension (d)...")
    d_values = [1, 2, 3, 5]
    m_fixed = 1000
    mse_d = []
    
    for d in d_values:
        X_train, _, Z_train, _ = generate_data(m_fixed, d=d, noise_type=noise_type)
        X_test, _, _, eta_test = generate_data(n_test, d=d, noise_type=noise_type)
        
        # Adaptation de la grille pour la dimension (h doit augmenter avec d)
        grid_d = np.linspace(0.1, 1.5, 15) * (d**0.25)
        selector = BandwidthSelector(h_grid=grid_d)
        h_cv, _ = selector.select_optimal_h(X_train, Z_train)
        
        model = EMNoiseEstimator(h_signal=h_cv, h_noise=h_cv * 1.5, max_iter=20)
        model.fit(X_train, Z_train)
        
        err = compute_mse(model, X_test, eta_test)
        mse_d.append(err)
        print(f"   d={d}: h_cv={h_cv:.3f} -> MSE={err:.5f}")

    # [3/3] Noise Level
    print("\n[3/3] Influence of Noise Level...")
    scales = [0.2, 0.5, 0.8, 1.0, 1.2]
    m_fixed_noise = 1000
    mse_noise = []
    
    for s in scales:
        X_train, _, Z_train, _ = generate_data(m_fixed_noise, d=1, noise_type=noise_type, noise_scale=s)
        X_test, _, _, eta_test = generate_data(n_test, d=1, noise_type=noise_type, noise_scale=s)
        
        selector = BandwidthSelector(h_grid=np.linspace(0.1, 1.0, 15))
        h_cv, _ = selector.select_optimal_h(X_train, Z_train)
        
        model = EMNoiseEstimator(h_signal=h_cv, h_noise=h_cv * 1.5, max_iter=20)
        model.fit(X_train, Z_train)
        
        err = compute_mse(model, X_test, eta_test)
        mse_noise.append(err)
        print(f"   Scale={s}: MSE={err:.5f}")

    # Plotting
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(m_values, mse_m, 'o-', color='blue', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Sample Size m')
    plt.ylabel('MSE')
    plt.title('Convergence Rate (CV)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(d_values, mse_d, 's-', color='red', lw=2)
    plt.xlabel('Dimension d')
    plt.title('Curse of Dimensionality (CV)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(scales, mse_noise, '^-', color='green', lw=2)
    plt.xlabel('Noise Scale')
    plt.title('Robustness (CV)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tests/performance_analysis_cv.png')
    print("\nAnalysis Saved: tests/performance_analysis_cv.png")
    plt.show()

if __name__ == "__main__":
    run_performance_suite()