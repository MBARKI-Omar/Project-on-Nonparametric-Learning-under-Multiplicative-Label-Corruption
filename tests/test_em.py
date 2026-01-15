"""
tests/test_em.py

Main visual test for the EM Estimator using Cross-Validation.
Displays both Linear and Sine noise cases side-by-side.
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.data_generation import generate_data, eta_function
from src.estimation_em import EMNoiseEstimator
from src.bandwidth_selection import BandwidthSelector

def run_test_em_cv():
    print("==================================================")
    print("TEST: EM Algorithm with Cross-Validation")
    print("==================================================")
    
    n_samples = 1500
    noise_types = ['linear', 'sine']
    
    plt.figure(figsize=(16, 6))
    
    for i, n_type in enumerate(noise_types):
        print(f"\n[{i+1}/2] Processing '{n_type}' noise...")
        
        # 1. Generate Data
        X, Y, Z, _ = generate_data(n_samples, d=1, noise_type=n_type, noise_scale=1.0)
        
        # 2. Find Optimal Bandwidth via LOOCV
        print("   -> Running LOOCV...")
        # Grille standard
        selector = BandwidthSelector(h_grid=np.linspace(0.1, 1.0, 20))
        h_cv, _ = selector.select_optimal_h(X, Z)
        print(f"   -> Selected h={h_cv:.3f}")
        
        # 3. Configure EM
        # On utilise h_cv pour le signal.
        # Pour le bruit, on élargit légèrement la fenêtre car l'estimation de variance 
        # nécessite plus de lissage que l'estimation de moyenne (ratio ~1.5 standard).
        h_sig = h_cv
        h_ns  = h_cv * 1.5
        
        print(f"   -> Fitting EM (h_sig={h_sig:.3f}, h_ns={h_ns:.3f})...")
        em_model = EMNoiseEstimator(h_signal=h_sig, h_noise=h_ns, max_iter=40)
        em_model.fit(X, Z)
        
        # 4. Predict
        X_grid = np.linspace(X.min(), X.max(), 400).reshape(-1, 1)
        eta_pred = em_model.predict_eta(X_grid)
        eta_true = eta_function(X_grid, noise_type=n_type)
        
        # 5. Plot
        plt.subplot(1, 2, i+1)
        plt.plot(X_grid, eta_true, 'g-', linewidth=3, label='True Noise')
        plt.plot(X_grid, eta_pred, 'r--', linewidth=3, label=f'EM (CV)')
        
        # Visualisation des flips réels
        flips = (Z != Y)
        if np.sum(flips) > 0:
            subset_idx = np.random.choice(np.where(flips)[0], size=min(100, np.sum(flips)), replace=False)
            plt.scatter(X[subset_idx], np.full(len(subset_idx), 0.01), 
                       color='black', marker='|', alpha=0.3)
        
        plt.title(f"Noise: {n_type.capitalize()}\n(CV: h={h_cv:.2f})")
        plt.xlabel("Feature X")
        plt.ylabel("Noise Level $\eta(x)$")
        plt.ylim(-0.02, 0.55)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'tests/results_em_cv_final.png'
    plt.savefig(output_file)
    print(f"\nGraph saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    run_test_em_cv()