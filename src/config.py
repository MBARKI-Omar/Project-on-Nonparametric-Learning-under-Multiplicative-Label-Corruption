"""
config.py

Centralized configuration for all experiments.
All hyperparameters and settings are defined here for reproducibility.

IMPORTANT: To ensure reproducibility when changing Q1 or Q2:
- Keep function names the same in data_generation.py and estimators.py
- Keep function signatures compatible (same parameters, same return format)
- Only modify function bodies
- If you need to change signatures, update USE_ADAPTIVE_WRAPPERS below
"""

import numpy as np


class ExperimentConfig:
    """
    Central configuration for all experiments.
    
    Modify parameters here to change experimental settings across all experiments.
    """
    
    # ========================================================================
    # REPRODUCIBILITY
    # ========================================================================
    random_seed = 42
    
    # ========================================================================
    # DATA GENERATION SETTINGS
    # ========================================================================
    # Default noise type for experiments
    default_noise_type = 'linear'
    
    # Available noise types
    available_noise_types = ['linear', 'sine']
    
    # ========================================================================
    # ESTIMATION SETTINGS
    # ========================================================================
    # Default kernel
    default_kernel = 'gaussian'
    
    # Available kernels
    available_kernels = ['gaussian', 'epanechnikov', 'tricube']
    
    # Bandwidth selection method ('silverman', 'cv', or manual value)
    bandwidth_method = 'silverman'
    
    # Default bandwidth (used if bandwidth_method is not 'silverman' or 'cv')
    default_bandwidth = 0.3
    
    # Bandwidth grid for cross-validation
    h_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    
    # ========================================================================
    # EXPERIMENT 1: BASIC ESTIMATION
    # ========================================================================
    exp1_sample_size = 1000
    exp1_dimension = 1
    exp1_noise_type = 'linear'
    exp1_n_test_points = 100  # Number of points to evaluate eta_hat
    
    # ========================================================================
    # EXPERIMENT 2: SAMPLE SIZE EFFECT
    # ========================================================================
    exp2_sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    exp2_dimension = 1
    exp2_noise_type = 'linear'
    exp2_n_trials = 20  # Monte Carlo trials
    exp2_n_test_points = 50
    
    # ========================================================================
    # EXPERIMENT 3: DIMENSION EFFECT
    # ========================================================================
    exp3_dimensions = [1, 2, 3, 5, 10]
    exp3_sample_size = 1000
    exp3_noise_type = 'linear'
    exp3_n_trials = 20
    exp3_n_test_points = 50
    
    # ========================================================================
    # EXPERIMENT 4: BANDWIDTH SELECTION
    # ========================================================================
    exp4_sample_size = 500
    exp4_dimension = 1
    exp4_noise_type = 'linear'
    exp4_n_test_points = 50
    
    # Part A: Sensitivity to h
    exp4a_h_values = np.array([0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    
    # Part B: Method comparison
    exp4b_methods = ['silverman', 'cv']
    exp4b_cv_h_grid = np.array([0.1, 0.2, 0.3, 0.5, 0.8])  # Smaller grid for speed
    
    # ========================================================================
    # EXPERIMENT 5: NOISE LEVEL EFFECT
    # ========================================================================
    exp5_sample_size = 1000
    exp5_dimension = 1
    exp5_noise_types = ['linear', 'sine']
    exp5_n_trials = 20
    exp5_n_test_points = 50
    
    # ========================================================================
    # VISUALIZATION SETTINGS
    # ========================================================================
    # Figure size
    figure_size = (10, 6)
    
    # DPI for saving figures
    figure_dpi = 300
    
    # Figure format
    figure_format = 'pdf'  # Can be 'pdf', 'png', or both
    
    # Matplotlib style
    plot_style = 'seaborn-v0_8-darkgrid'
    
    # Font sizes
    font_size_title = 14
    font_size_labels = 12
    font_size_legend = 10
    
    # Colors
    color_true = '#2E86AB'      # Blue for true values
    color_estimated = '#A23B72'  # Purple for estimated values
    color_error = '#F18F01'      # Orange for errors
    
    # ========================================================================
    # FILE PATHS
    # ========================================================================
    figures_dir = 'figures/'
    results_dir = 'results/'
    
    # ========================================================================
    # ADVANCED: ADAPTIVE WRAPPERS (for changing function signatures)
    # ========================================================================
    # Set to True if you change function signatures in data_generation.py or estimators.py
    # This enables automatic detection and adaptation
    # Default: False (assumes signatures remain compatible)
    use_adaptive_wrappers = False
    
    # If your new estimation method needs clean labels, specify the ratio
    clean_label_ratio = 0.1  # 10% of data will be treated as clean
    
    # ========================================================================
    # MONTE CARLO SETTINGS
    # ========================================================================
    # Default number of trials for experiments with multiple runs
    default_n_trials = 20
    
    # Confidence level for error bars
    confidence_level = 0.95


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config():
    """
    Get the experiment configuration.
    
    Returns:
        ExperimentConfig: Configuration object
    """
    return ExperimentConfig()


def print_config():
    """Print current configuration."""
    config = ExperimentConfig()
    
    print("="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"\nRandom Seed: {config.random_seed}")
    print(f"Default Noise Type: {config.default_noise_type}")
    print(f"Default Kernel: {config.default_kernel}")
    print(f"Bandwidth Method: {config.bandwidth_method}")
    
    print("\nExperiment Settings:")
    print(f"  Exp 1 - Sample Size: {config.exp1_sample_size}")
    print(f"  Exp 2 - Sample Sizes: {config.exp2_sample_sizes}")
    print(f"  Exp 2 - Trials: {config.exp2_n_trials}")
    print(f"  Exp 3 - Dimensions: {config.exp3_dimensions}")
    print(f"  Exp 3 - Trials: {config.exp3_n_trials}")
    
    print("\nOutput Directories:")
    print(f"  Figures: {config.figures_dir}")
    print(f"  Results: {config.results_dir}")
    
    print("="*70)


if __name__ == "__main__":
    print_config()