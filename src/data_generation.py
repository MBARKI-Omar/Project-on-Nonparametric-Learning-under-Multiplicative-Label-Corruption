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
    Generate true labels Y with perfect separability.
    
    Uses random assignment: each observation is randomly assigned to class +1 or -1.
    
    Args:
        X: array of shape (n, d)
        
    Returns:
        Y: array of shape (n,) with values in {-1, 1}
    """
    n = X.shape[0]
    
    # Randomly assign each observation to class +1 or -1 with probability 0.5 each
    # This creates a deterministic mapping X -> Y while maintaining balance
    Y = np.random.choice([-1, 1], size=n)
    
    return Y


def generate_Z(X, Y, noise_type='linear'):
    """
    Generate corrupted labels Z from true labels Y according to the noise model.
    
    Corruption model:
        P(Z = -Y | X, Y) = η(X)  (flip with probability η)
        P(Z = Y | X, Y) = 1 - η(X)  (keep with probability 1-η)
    
    Args:
        X: array of shape (n, d) - features
        Y: array of shape (n,) - true labels {-1, 1}
        noise_type: type of noise function ('linear', 'sine')
        
    Returns:
        Z: array of shape (n,) - corrupted labels {-1, 1}
        eta_values: array of shape (n,) - true noise values η(X)
    """
    # Compute the noise level η(X) for each observation
    eta_values = eta_function(X, noise_type=noise_type)
    
    # Generate random uniform numbers in [0, 1] for each observation
    # This will determine which labels get flipped
    random_flips = np.random.rand(X.shape[0])
    
    # Create a boolean mask: True where we should flip the label
    # flip_mask[i] = True if random_flips[i] < η(X[i])
    flip_mask = random_flips < eta_values
    
    # Start with Z = Y (copy the true labels)
    Z = Y.copy()
    
    # Flip the labels where flip_mask is True
    # Since Y ∈ {-1, 1}, flipping means multiplying by -1
    Z[flip_mask] = -Z[flip_mask]
    
    return Z, eta_values


def eta_function(X, noise_type='linear'):
    """
    Compute the noise function η(X).
    
    Args:
        X: array of shape (n, d)
        noise_type: type of noise ('linear' or 'sine')
        
    Returns:
        eta: array of shape (n,) with values in [0, 0.5[
    """
    if noise_type == 'linear':
        # Linear noise: η varies from 0.0 to 0.4 based on X[0]
        # Normalize X[0] to [0, 1]
        x_normalized = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
        # Scale to [0, 0.4]
        return 0.4 * x_normalized
    
    elif noise_type == 'sine':
        # Sinusoidal noise: η(x) = 0.2 + 0.15 * sin(2 * X[0])
        eta = 0.2 + 0.15 * np.sin(2 * X[:, 0])
        # Clip to ensure η ∈ [0, 0.5[
        return np.clip(eta, 0, 0.49)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Use 'linear' or 'sine'.")


def generate_data(n, d=1, noise_type='linear'):
    """
    Generate a complete dataset (X, Y, Z, eta_true).
    
    This is a convenience function that combines all generation steps.
    
    Args:
        n: number of observations
        d: dimension of feature space
        noise_type: type of noise function ('linear', 'sine')
        
    Returns:
        X: array of shape (n, d) - features
        Y: array of shape (n,) - true labels {-1, 1}
        Z: array of shape (n,) - corrupted labels {-1, 1}
        eta_true: array of shape (n,) - true noise values η(X)
    """
    # Step 1: Generate features X ~ N(0, I_d)
    X = generate_X(n, d)
    
    # Step 2: Generate true labels Y ∈ {-1, 1} (deterministic given X)
    Y = generate_Y(X)
    
    # Step 3: Generate corrupted labels Z by flipping Y with probability η(X)
    Z, eta_true = generate_Z(X, Y, noise_type=noise_type)
    
    return X, Y, Z, eta_true