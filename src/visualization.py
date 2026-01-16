"""
visualization.py

Plotting functions for creating publication-ready figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import os


def setup_plot_style(config):
    """
    Setup matplotlib style from config.
    
    Args:
        config: ExperimentConfig object
    """
    try:
        plt.style.use(config.plot_style)
    except:
        plt.style.use('default')
    
    plt.rcParams['font.size'] = config.font_size_labels
    plt.rcParams['axes.titlesize'] = config.font_size_title
    plt.rcParams['axes.labelsize'] = config.font_size_labels
    plt.rcParams['legend.fontsize'] = config.font_size_legend


def save_figure(fig, filename: str, config, tight: bool = True):
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename (without extension)
        config: ExperimentConfig object
        tight: Use tight layout
    """
    if tight:
        fig.tight_layout()
    
    # Create directory if needed
    os.makedirs(config.figures_dir, exist_ok=True)
    
    # Save figure
    filepath = os.path.join(config.figures_dir, f"{filename}.{config.figure_format}")
    fig.savefig(filepath, dpi=config.figure_dpi, bbox_inches='tight')
    print(f"Figure saved: {filepath}")


def plot_eta_1d(x: np.ndarray, 
                eta_true: np.ndarray, 
                eta_hat: np.ndarray,
                config,
                title: str = "Noise Function Estimation",
                filename: str = "eta_estimation_1d"):
    """
    Plot true vs estimated noise function for 1D case.
    
    Args:
        x: Input points of shape (n,)
        eta_true: True noise values of shape (n,)
        eta_hat: Estimated noise values of shape (n,)
        config: ExperimentConfig object
        title: Plot title
        filename: Output filename
    """
    setup_plot_style(config)
    
    fig, ax = plt.subplots(figsize=config.figure_size)
    
    # Sort by x for plotting
    idx = np.argsort(x)
    x_sorted = x[idx]
    eta_true_sorted = eta_true[idx]
    eta_hat_sorted = eta_hat[idx]
    
    # Plot true and estimated
    ax.plot(x_sorted, eta_true_sorted, 'o-', 
            color=config.color_true, linewidth=2, markersize=4,
            label='True η(x)', alpha=0.8)
    ax.plot(x_sorted, eta_hat_sorted, 's--', 
            color=config.color_estimated, linewidth=2, markersize=4,
            label='Estimated η̂(x)', alpha=0.8)
    
    # Shaded error region
    ax.fill_between(x_sorted, eta_true_sorted, eta_hat_sorted, 
                    color=config.color_error, alpha=0.2, label='Error')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Noise Level η(x)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 0.55])
    
    save_figure(fig, filename, config)
    plt.close(fig)


def plot_mse_vs_sample_size(m_values: np.ndarray,
                            mse_means: np.ndarray,
                            mse_stds: np.ndarray,
                            config,
                            d: int = 1,
                            title: str = "Convergence with Sample Size",
                            filename: str = "convergence_sample_size"):
    """
    Plot MSE vs sample size in log-log scale with theoretical rate.
    
    Args:
        m_values: Sample sizes
        mse_means: Mean MSE values
        mse_stds: Standard deviations of MSE
        config: ExperimentConfig object
        d: Dimension (for theoretical rate)
        title: Plot title
        filename: Output filename
    """
    setup_plot_style(config)
    
    fig, ax = plt.subplots(figsize=config.figure_size)
    
    # Plot empirical MSE
    ax.errorbar(m_values, mse_means, yerr=mse_stds, 
                fmt='o-', color=config.color_estimated, 
                linewidth=2, markersize=8, capsize=5,
                label='Empirical MSE', alpha=0.8)
    
    # Plot theoretical rate: MSE ~ m^{-4/(d+4)}
    rate = -4 / (d + 4)
    # Fit constant to match empirical curve
    C = mse_means[len(m_values)//2] / (m_values[len(m_values)//2] ** rate)
    theoretical = C * m_values ** rate
    
    ax.plot(m_values, theoretical, '--', 
            color=config.color_true, linewidth=2,
            label=f'Theoretical rate: m^{{{rate:.2f}}}', alpha=0.7)
    
    ax.set_xlabel('Sample Size (m)')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    save_figure(fig, filename, config)
    plt.close(fig)


def plot_mse_vs_dimension(d_values: np.ndarray,
                         mse_means: np.ndarray,
                         mse_stds: np.ndarray,
                         config,
                         m: int = 1000,
                         title: str = "Curse of Dimensionality",
                         filename: str = "curse_dimensionality"):
    """
    Plot MSE vs dimension (barplot).
    
    Args:
        d_values: Dimensions
        mse_means: Mean MSE values
        mse_stds: Standard deviations of MSE
        config: ExperimentConfig object
        m: Sample size (for theoretical rate)
        title: Plot title
        filename: Output filename
    """
    setup_plot_style(config)
    
    fig, ax = plt.subplots(figsize=config.figure_size)
    
    # Barplot with error bars
    x_pos = np.arange(len(d_values))
    bars = ax.bar(x_pos, mse_means, yerr=mse_stds, 
                  color=config.color_estimated, alpha=0.7,
                  capsize=5, edgecolor='black', linewidth=1.5)
    
    # Add theoretical convergence rates as text
    for i, (d, mse_mean) in enumerate(zip(d_values, mse_means)):
        rate = -4 / (d + 4)
        ax.text(i, mse_mean + mse_stds[i] + 0.005, 
                f'Rate: {rate:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Dimension (d)')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'd={d}' for d in d_values])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add note about sample size
    ax.text(0.02, 0.98, f'Sample size: m={m}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_figure(fig, filename, config)
    plt.close(fig)


def plot_mse_vs_bandwidth(h_values: np.ndarray,
                         mse_values: np.ndarray,
                         h_silverman: Optional[float] = None,
                         h_cv: Optional[float] = None,
                         config = None,
                         title: str = "Bandwidth Selection",
                         filename: str = "bandwidth_sensitivity"):
    """
    Plot MSE vs bandwidth (U-shape curve).
    
    Args:
        h_values: Bandwidth values
        mse_values: Corresponding MSE values
        h_silverman: Silverman's bandwidth (optional, to mark on plot)
        h_cv: CV-selected bandwidth (optional, to mark on plot)
        config: ExperimentConfig object
        title: Plot title
        filename: Output filename
    """
    setup_plot_style(config)
    
    fig, ax = plt.subplots(figsize=config.figure_size)
    
    # Plot MSE curve
    ax.plot(h_values, mse_values, 'o-', 
            color=config.color_estimated, linewidth=2, 
            markersize=8, label='MSE(h)')
    
    # Mark optimal h
    h_opt_idx = np.argmin(mse_values)
    h_opt = h_values[h_opt_idx]
    mse_opt = mse_values[h_opt_idx]
    ax.plot(h_opt, mse_opt, '*', color='red', 
            markersize=20, label=f'Optimal: h={h_opt:.3f}')
    
    # Mark Silverman's h
    if h_silverman is not None:
        ax.axvline(h_silverman, color=config.color_true, 
                  linestyle='--', linewidth=2, 
                  label=f"Silverman's rule: h={h_silverman:.3f}")
    
    # Mark CV h
    if h_cv is not None:
        ax.axvline(h_cv, color='green', 
                  linestyle=':', linewidth=2, 
                  label=f'Cross-validation: h={h_cv:.3f}')
    
    ax.set_xlabel('Bandwidth (h)')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, filename, config)
    plt.close(fig)


def plot_method_comparison(methods: List[str],
                          mse_values: List[float],
                          h_values: List[float],
                          config,
                          title: str = "Bandwidth Selection Method Comparison",
                          filename: str = "method_comparison"):
    """
    Compare different bandwidth selection methods.
    
    Args:
        methods: List of method names
        mse_values: MSE for each method
        h_values: Selected h for each method
        config: ExperimentConfig object
        title: Plot title
        filename: Output filename
    """
    setup_plot_style(config)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x_pos = np.arange(len(methods))
    colors = [config.color_true, config.color_estimated, 'green'][:len(methods)]
    
    # Plot MSE comparison
    bars1 = ax1.bar(x_pos, mse_values, color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.set_title('MSE Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, mse) in enumerate(zip(bars1, mse_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot selected h values
    bars2 = ax2.bar(x_pos, h_values, color=colors, alpha=0.7,
                    edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Selected Bandwidth (h)')
    ax2.set_title('Bandwidth Selection')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, h) in enumerate(zip(bars2, h_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{h:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    fig.suptitle(title, fontsize=config.font_size_title)
    
    save_figure(fig, filename, config)
    plt.close(fig)


def plot_noise_robustness(noise_info: List[dict],
                         mse_values: List[float],
                         mse_stds: List[float],
                         config,
                         title: str = "Robustness to Noise Level",
                         filename: str = "noise_robustness"):
    """
    Plot MSE vs noise level/type.
    
    Args:
        noise_info: List of dicts with 'type' and 'mean_eta'
        mse_values: MSE values
        mse_stds: Standard deviations
        config: ExperimentConfig object
        title: Plot title
        filename: Output filename
    """
    setup_plot_style(config)
    
    fig, ax = plt.subplots(figsize=config.figure_size)
    
    # Extract info
    labels = [f"{info['type']}\n(η̄={info['mean_eta']:.2f})" 
              for info in noise_info]
    x_pos = np.arange(len(labels))
    
    # Create color map by noise type
    colors = []
    for info in noise_info:
        if info['type'] == 'linear':
            colors.append(config.color_true)
        elif info['type'] == 'sine':
            colors.append(config.color_estimated)
        else:
            colors.append(config.color_error)
    
    # Barplot
    bars = ax.bar(x_pos, mse_values, yerr=mse_stds,
                  color=colors, alpha=0.7, capsize=5,
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Noise Configuration')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for noise types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=config.color_true, alpha=0.7, label='Linear noise'),
        Patch(facecolor=config.color_estimated, alpha=0.7, label='Sine noise')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    save_figure(fig, filename, config)
    plt.close(fig)


if __name__ == "__main__":
    # Test visualization module
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    
    print("Testing visualization module...")
    print("Generating sample plots...\n")
    
    # Test 1D plot
    x = np.linspace(-2, 2, 50)
    eta_true = 0.2 + 0.1 * np.sin(2 * x)
    eta_hat = eta_true + np.random.randn(50) * 0.03
    
    plot_eta_1d(x, eta_true, eta_hat, config, 
                title="Test: 1D Estimation",
                filename="test_eta_1d")
    
    # Test convergence plot
    m_values = np.array([50, 100, 200, 500, 1000])
    mse_means = 0.5 * m_values ** (-0.8) + np.random.randn(5) * 0.01
    mse_stds = mse_means * 0.1
    
    plot_mse_vs_sample_size(m_values, mse_means, mse_stds, config,
                           filename="test_convergence")
    
    print("\nTest plots saved in", config.figures_dir)