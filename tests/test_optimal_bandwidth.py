"""
tests/test_optimal_bandwidth.py

Validates the BandwidthSelector by checking the convergence rate.
Theory: h_opt ~ m^(-1/(d+4)). For d=1, slope should be approx -0.2.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from src.data_generation import generate_data
from src.bandwidth_selection import BandwidthSelector

def run_bandwidth_verification():
    print("==================================================")
    print("VERIFICATION: LOOCV Convergence Rate")
    print("==================================================")

    m_values = [100, 200, 400, 800, 1500]
    optimal_hs = []
    
    # Grid fine pour la précision du test théorique
    h_grid_search = np.logspace(np.log10(0.05), np.log10(1.0), 50)
    
    print("Finding optimal bandwidths for increasing sample sizes...")
    
    for m in m_values:
        # Generate Data (Sine noise is standard for testing complexity)
        X, _, Z, _ = generate_data(m, d=1, noise_type='sine', noise_scale=1.0)
        
        selector = BandwidthSelector(h_grid=h_grid_search)
        best_h, min_err = selector.select_optimal_h(X, Z)
        
        optimal_hs.append(best_h)
        print(f"   m={m}: Best h={best_h:.3f} (CV Error={min_err:.4f})")

    # Linear Regression in Log-Log space
    log_m = np.log(m_values)
    log_h = np.log(optimal_hs)
    
    slope, intercept, _, _, _ = linregress(log_m, log_h)
    
    print(f"\n   >>> Calculated Slope: {slope:.4f}")
    print(f"   >>> Theoretical Slope (d=1): -0.2000")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(log_m, log_h, color='red', s=100, label='Empirical LOOCV')
    plt.plot(log_m, intercept + slope * log_m, 'b--', linewidth=2, 
             label=f'Fit: slope={slope:.2f}')
    plt.plot(log_m, intercept + (-0.2) * log_m, 'g:', linewidth=2, 
             label='Theory: slope=-0.2')
    
    plt.xlabel('log(Sample Size m)')
    plt.ylabel('log(Optimal Bandwidth h)')
    plt.title('Convergence of Optimal Bandwidth (LOOCV)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_file = 'tests/optimal_bandwidth_check.png'
    plt.savefig(output_file)
    print(f"Verification plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    run_bandwidth_verification()