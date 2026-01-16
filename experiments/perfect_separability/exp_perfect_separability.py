"""
Debug: Test direct sans imports compliqués
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# TOUT EN UN FICHIER - AUCUN IMPORT
# ============================================================================

def gaussian_kernel(distances):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * distances**2)

def silverman_bandwidth(X):
    m, d = X.shape
    std = np.std(X, axis=0).mean()
    h = ((4 / (d + 2)) ** (1 / (d + 4))) * std * (m ** (-1 / (d + 4)))
    return h

def nadaraya_watson(X_train, Z_train, X_test, h):
    n_test = X_test.shape[0]
    r_hat = np.zeros(n_test)
    
    for i in range(n_test):
        diff = (X_train - X_test[i]) / h
        distances = np.linalg.norm(diff, axis=1)
        weights = gaussian_kernel(distances)
        sum_weights = np.sum(weights)
        
        if sum_weights > 1e-10:
            r_hat[i] = np.sum(weights * Z_train) / sum_weights
        else:
            r_hat[i] = np.mean(Z_train)
    
    return r_hat

def estimate_eta(X_train, Z_train, X_test, h=None):
    if h is None:
        h = silverman_bandwidth(X_train)
    
    print(f"  Using h = {h:.6f}")
    
    r_hat = nadaraya_watson(X_train, Z_train, X_test, h)
    eta_hat = (1 - np.abs(r_hat)) / 2
    eta_hat = np.clip(eta_hat, 0, 0.499)
    
    return eta_hat

def eta_function(X, noise_type='linear'):
    """True noise function."""
    if X.shape[1] == 1:
        x = X[:, 0]
    else:
        x = X[:, 0]  # Use first dimension
    
    if noise_type == 'linear':
        return 0.2 + 0.25 * x
    else:  # sine
        return 0.2 + 0.15 * np.sin(np.pi * x)

# ============================================================================
# GENERATE DATA
# ============================================================================

def generate_deterministic(m, d):
    X = np.random.uniform(0, 1, size=(m, d))
    
    # DETERMINISTIC Y
    if d == 1:
        Y = np.where(X[:, 0] > 0.5, 1, -1)
    else:
        Y = np.where(X[:, 0] + X[:, 1] > 1.0, 1, -1)
    
    # True noise
    eta_vals = eta_function(X, 'linear')
    
    # Corrupt
    flip = np.random.rand(m) < eta_vals
    Z = np.where(flip, -Y, Y)
    
    return X, Y, Z, eta_vals

def generate_boundary(m, d):
    X = np.random.uniform(0, 1, size=(m, d))
    
    # PROBABILISTIC Y
    if d == 1:
        z = 3 * (X[:, 0] - 0.5)
    else:
        z = 2 * (X[:, 0] + X[:, 1] - 1.0)
    
    probs = 1 / (1 + np.exp(-z))
    Y = np.where(np.random.rand(m) < probs, 1, -1)
    
    eta_vals = eta_function(X, 'linear')
    flip = np.random.rand(m) < eta_vals
    Z = np.where(flip, -Y, Y)
    
    return X, Y, Z, eta_vals

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

print("=" * 70)
print("DEBUG EXPERIMENT")
print("=" * 70)

m, d = 2000, 2

# Scenario 1
print("\n[Scenario 1] Deterministic")
X1, Y1, Z1, eta1 = generate_deterministic(m, d)
eta_pred1 = estimate_eta(X1, Z1, X1, h=None)
mse1 = np.mean((eta_pred1 - eta1)**2)
print(f"  MSE = {mse1:.6f}")

# Scenario 2
print("\n[Scenario 2] Boundary")
X2, Y2, Z2, eta2 = generate_boundary(m, d)
eta_pred2 = estimate_eta(X2, Z2, X2, h=None)
mse2 = np.mean((eta_pred2 - eta2)**2)
print(f"  MSE = {mse2:.6f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

X_grid = np.column_stack([np.linspace(0, 1, 200), 0.5 * np.ones(200)])
eta_grid = eta_function(X_grid, 'linear')

eta_pred_grid1 = estimate_eta(X1, Z1, X_grid, h=None)
eta_pred_grid2 = estimate_eta(X2, Z2, X_grid, h=None)

# Plot 1
axes[0, 0].plot(X_grid[:, 0], eta_grid, 'b-', linewidth=3, label='True', alpha=0.8)
axes[0, 0].plot(X_grid[:, 0], eta_pred_grid1, 'go', markersize=4, label='Pred', alpha=0.6)
axes[0, 0].set_title(f'Scenario 1: MSE={mse1:.6f}', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_ylim([0, 0.5])

# Plot 2
axes[0, 1].scatter(eta1, eta_pred1, alpha=0.3, s=10, c='steelblue')
axes[0, 1].plot([0, 0.5], [0, 0.5], 'k--', linewidth=2)
axes[0, 1].set_title('Scenario 1: Quality')
axes[0, 1].set_aspect('equal')
axes[0, 1].grid(alpha=0.3)

# Plot 3
axes[1, 0].plot(X_grid[:, 0], eta_grid, 'b-', linewidth=3, label='True', alpha=0.8)
axes[1, 0].plot(X_grid[:, 0], eta_pred_grid2, 'ro', markersize=4, label='Pred', alpha=0.6)
axes[1, 0].set_title(f'Scenario 2: MSE={mse2:.6f}', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_ylim([0, 0.5])

# Plot 4
axes[1, 1].scatter(eta2, eta_pred2, alpha=0.3, s=10, c='indianred')
axes[1, 1].plot([0, 0.5], [0, 0.5], 'k--', linewidth=2)
axes[1, 1].set_title('Scenario 2: Quality')
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(alpha=0.3)

plt.suptitle('DEBUG: Perfect Separability Test', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('debug_result.pdf', dpi=300, bbox_inches='tight')
print("\nSaved: debug_result.pdf")
plt.show()

print(f"\nDegradation: {mse2/mse1:.1f}x")
print("=" * 70)