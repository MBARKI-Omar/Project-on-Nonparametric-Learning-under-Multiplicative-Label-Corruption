"""
data_generation.py

Synthetic data generation for the label corruption project.
Assumption: Perfect separability (P(Y=1|X) ∈ {0, 1})
"""

import numpy as np


def generate_X(n, d=1):
    """
    Generate feature matrix X.
    
    Args:
        n: number of observations
        d: dimension of feature space
        
    Returns:
        X: array of shape (n, d) ~ N(0, I_d)
    """
    # Generate X from standard multivariate normal distribution
    X = np.random.randn(n, d)
    
    return X


def generate_Y(X):
    """
    Generate true labels Y with perfect separability based on feature values.
    
    Unlike random assignment, this creates a learnable structure:
    Y = sign(X[0])
    
    Args:
        X: array of shape (n, d)
        
    Returns:
        Y: array of shape (n,) with values in {-1, 1}
    """
    # Create a linear decision boundary based on the first feature
    # If X[0] > 0, Y = 1, else Y = -1
    Y = np.sign(X[:, 0])
    
    # Handle the rare case where X[0] is exactly 0
    Y[Y == 0] = 1
    
    return Y


def eta_function(X, noise_type='linear', noise_scale=1.0):
    """
    Compute the label corruption probability η(x) with a scaling factor.
    
    The function models the probability that a true label Y is flipped to Z,
    which depends on the input features. Two noise models are available:
    - 'linear': Linear ramp normalized to [0, 1], scaled by 0.4
    - 'sine': Sinusoidal pattern with mean 0.2 and amplitude 0.15
    
    Args:
        X: Feature matrix of shape (n, d)
        noise_type: Type of noise model ('linear' or 'sine')
        noise_scale: Scaling factor for noise intensity (multiplicative)
        
    Returns:
        eta: Array of shape (n,) with corruption probabilities clipped to [0, 0.499]
             Values are capped below 0.5 to maintain label separability
    """
    if noise_type == 'linear':
        # Normalized linear ramp: maps X[:, 0] from [-3, 3] to [0, 1]
        val = (X[:, 0] + 3) / 6
        val = np.clip(val, 0, 1)
        base_eta = 0.4 * val
    
    elif noise_type == 'sine':
        # Sinusoidal corruption: η(x) = 0.2 + 0.15 * sin(2x)
        base_eta = 0.2 + 0.15 * np.sin(2 * X[:, 0])
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Apply scaling factor and clip to [0, 0.499] to preserve label identifiability
    return np.clip(base_eta * noise_scale, 0, 0.499)


def generate_Z(X, Y, noise_type='linear', noise_scale=1.0):
    """
    Generate corrupted labels Z by flipping true labels Y according to η(x).
    
    Args:
        X: Feature matrix of shape (n, d)
        Y: True labels of shape (n,) with values in {-1, 1}
        noise_type: Type of corruption model
        noise_scale: Scaling factor for corruption intensity
        
    Returns:
        Z: Corrupted labels of shape (n,) with values in {-1, 1}
        eta_values: Corruption probabilities of shape (n,)
    """
    eta_values = eta_function(X, noise_type=noise_type, noise_scale=noise_scale)
    
    # Generate random flips with probability η(x) for each sample
    random_flips = np.random.rand(X.shape[0])
    flip_mask = random_flips < eta_values
    
    # Copy true labels and flip those selected by the mask
    Z = Y.copy()
    Z[flip_mask] = -Z[flip_mask]
    
    return Z, eta_values


def generate_data(n, d=1, noise_type='linear', noise_scale=1.0):
    """
    Generate complete synthetic dataset with features, true labels, and corrupted labels.
    
    Args:
        n: Number of observations
        d: Dimension of feature space (default: 1)
        noise_type: Type of corruption model ('linear' or 'sine')
        noise_scale: Scaling factor controlling corruption intensity
        
    Returns:
        X: Feature matrix of shape (n, d)
        Y: True labels of shape (n,)
        Z: Corrupted labels of shape (n,)
        eta_true: True corruption probabilities of shape (n,)
    """
    X = generate_X(n, d)
    Y = generate_Y(X)
    Z, eta_true = generate_Z(X, Y, noise_type=noise_type, noise_scale=noise_scale)
    
    return X, Y, Z, eta_true