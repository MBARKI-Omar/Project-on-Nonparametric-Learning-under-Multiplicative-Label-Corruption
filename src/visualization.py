"""
Visualization functions for noise estimation experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_eta_1d(X, eta_true, eta_pred, save_path=None):
    """
    Plot 1D comparison of true vs predicted noise.
    
    Args:
        X: features, shape (n, 1) or (n,)
        eta_true: true noise, shape (n,)
        eta_pred: predicted noise, shape (n,), may contain NaN
        save_path: optional path to save figure
    """
    if X.ndim == 2:
        X = X[:, 0]
    
    # Sort by X for nice plotting
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    eta_true_sorted = eta_true[sort_idx]
    eta_pred_sorted = eta_pred[sort_idx]
    
    # Separate identifiable and non-identifiable
    mask = ~np.isnan(eta_pred_sorted)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot true function
    ax.plot(X_sorted, eta_true_sorted, 'b-', linewidth=2, label='True η(x)', alpha=0.7)
    
    # Plot predictions (identifiable points)
    ax.scatter(X_sorted[mask], eta_pred_sorted[mask], 
              c='green', s=20, alpha=0.6, label='Predicted η(x)', zorder=3)
    
    # Mark non-identifiable regions
    ax.scatter(X_sorted[~mask], eta_true_sorted[~mask],
              c='red', s=20, alpha=0.3, label='Non-identifiable', marker='x')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('η(x)', fontsize=12)
    ax.set_title('Noise Estimation: True vs Predicted (1D)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_eta_2d(X_grid, eta_true_grid, eta_pred_grid, r_hat_grid=None, save_path=None):
    """
    Plot 2D heatmaps: true eta, predicted eta, |r_hat|, confidence map.
    
    Args:
        X_grid: grid points, shape (n*n, 2)
        eta_true_grid: true noise on grid, shape (n, n)
        eta_pred_grid: predicted noise on grid, shape (n, n), may contain NaN
        r_hat_grid: optional regression values, shape (n, n)
        save_path: optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Determine grid extent
    x1_min, x1_max = X_grid[:, 0].min(), X_grid[:, 0].max()
    x2_min, x2_max = X_grid[:, 1].min(), X_grid[:, 1].max()
    extent = [x1_min, x1_max, x2_min, x2_max]
    
    # Plot 1: True eta
    im1 = axes[0, 0].imshow(eta_true_grid, origin='lower', extent=extent, 
                            cmap='viridis', aspect='auto')
    axes[0, 0].set_title('True η(x)', fontsize=12)
    axes[0, 0].set_xlabel('x₁')
    axes[0, 0].set_ylabel('x₂')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Predicted eta (NaN shown as white)
    eta_pred_plot = np.copy(eta_pred_grid)
    im2 = axes[0, 1].imshow(eta_pred_plot, origin='lower', extent=extent,
                            cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Predicted η̂(x)', fontsize=12)
    axes[0, 1].set_xlabel('x₁')
    axes[0, 1].set_ylabel('x₂')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: |r_hat| if provided, else confidence
    if r_hat_grid is not None:
        im3 = axes[1, 0].imshow(np.abs(r_hat_grid), origin='lower', extent=extent,
                                cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_title('|r̂(x)| = |E[Z|X]|', fontsize=12)
    else:
        im3 = axes[1, 0].imshow(~np.isnan(eta_pred_grid), origin='lower', extent=extent,
                                cmap='RdYlGn', aspect='auto')
        axes[1, 0].set_title('Identifiable Regions', fontsize=12)
    axes[1, 0].set_xlabel('x₁')
    axes[1, 0].set_ylabel('x₂')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot 4: Confidence map (binary)
    confidence = ~np.isnan(eta_pred_grid)
    im4 = axes[1, 1].imshow(confidence, origin='lower', extent=extent,
                            cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_title('Confidence Map', fontsize=12)
    axes[1, 1].set_xlabel('x₁')
    axes[1, 1].set_ylabel('x₂')
    cbar = plt.colorbar(im4, ax=axes[1, 1], ticks=[0, 1])
    cbar.ax.set_yticklabels(['Non-identifiable', 'Identifiable'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_bandwidth_effect(h_values, metrics_list, save_path=None):
    """
    Plot MSE vs bandwidth h.
    
    Args:
        h_values: list of bandwidth values
        metrics_list: list of metric dicts
        save_path: optional path to save figure
    """
    mse_values = [m['mse'] for m in metrics_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(h_values, mse_values, 'bo-', linewidth=2, markersize=8)
    
    # Mark optimal h
    opt_idx = np.argmin(mse_values)
    ax.plot(h_values[opt_idx], mse_values[opt_idx], 'r*', 
            markersize=20, label=f'Optimal h = {h_values[opt_idx]:.3f}')
    
    ax.set_xlabel('Bandwidth h', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Effect of Bandwidth on Estimation Error', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sample_size_effect(m_values, metrics_list, save_path=None):
    """
    Plot MSE vs sample size m in log-log scale.
    Fit linear regression to estimate empirical convergence rate.
    
    Args:
        m_values: list of sample sizes
        metrics_list: list of metric dicts
        save_path: optional path to save figure
    """
    mse_values = [m['mse'] for m in metrics_list]
    
    # Log-log scale
    log_m = np.log(m_values)
    log_mse = np.log(mse_values)
    
    # Fit linear regression
    slope, intercept = np.polyfit(log_m, log_mse, 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    ax.loglog(m_values, mse_values, 'bo', markersize=8, label='Empirical MSE')
    
    # Plot fitted line
    m_fit = np.linspace(min(m_values), max(m_values), 100)
    mse_fit = np.exp(intercept) * m_fit ** slope
    ax.loglog(m_fit, mse_fit, 'r--', linewidth=2, 
             label=f'Fit: MSE ∝ m^{slope:.2f}')
    
    # Theoretical rate for d=2
    theoretical_slope = -4/(2+4)  # -0.67
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.1)
    
    ax.set_xlabel('Sample size m', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title(f'Convergence Rate: Empirical {slope:.2f} vs Theoretical {theoretical_slope:.2f}', 
                fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_dimension_effect(d_values, metrics_list, save_path=None):
    """
    Plot MSE vs dimension d (curse of dimensionality).
    
    Args:
        d_values: list of dimensions
        metrics_list: list of metric dicts
        save_path: optional path to save figure
    """
    mse_values = [m['mse'] for m in metrics_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(d_values, mse_values, color='steelblue', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Dimension d', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Curse of Dimensionality: MSE vs Dimension', fontsize=14)
    ax.set_xticks(d_values)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_noise_level_effect(noise_levels, metrics_list, save_path=None):
    """
    Plot MSE vs noise level (low, medium, high).
    
    Args:
        noise_levels: list of noise level strings ['low', 'medium', 'high']
        metrics_list: list of metric dicts
        save_path: optional path to save figure
    """
    mse_values = [m['mse'] for m in metrics_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green', 'orange', 'red']
    ax.bar(noise_levels, mse_values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Effect of Noise Level on Estimation Error', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()