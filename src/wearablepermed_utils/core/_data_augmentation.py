import numpy as np
from scipy.interpolate import CubicSpline

# Jittering

def jitter(X, sigma=0.5):
    """
    Add Gaussian noise to the data.

    Parameters:
    X (numpy.ndarray): Input data array.
    sigma (float): Standard deviation of the Gaussian noise.

    Returns:
    numpy.ndarray: Data array with added noise.
    """
    return X + np.random.normal(loc=0, scale=sigma, size=X.shape)

# Magnitude Warping
def magnitude_warp(X, sigma=0.2):
    """
    Apply magnitude distortion to the data.

    Parameters:
    X (numpy.ndarray): Input data array.
    sigma (float): Standard deviation of the magnitude distortion.

    Returns:
    numpy.ndarray: Data array with magnitude distortion applied.
    """
    return X * (np.random.normal(1, sigma, (X.shape[0], 1, X.shape[2])))

# Shifting
def time_shift(X, shift_max=2):
    """
    Apply temporal shifting to the data.

    Parameters:
    X (numpy.ndarray): Input data array.
    shift_max (int): Maximum number of positions to shift.

    Returns:
    numpy.ndarray: Temporally shifted data array.
    """
    
    if(shift_max == 0):
        return X.copy()
    
    X_new = np.zeros_like(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(-shift_max, shift_max)
        X_new[i] = np.roll(X[i], shift, axis=0)
    return X_new

# Scaling
def scale(X, sigma=0.1):
    """
    Apply random scaling to the data.

    Parameters:
    X (numpy.ndarray): Input data array.
    sigma (float): Standard deviation of the scaling factor.

    Returns:
    numpy.ndarray: Scaled data array.
    """
    scaling_factor = np.random.normal(1, sigma, (X.shape[0], 1, X.shape[2]))
    return X * scaling_factor

# Time Warping
def time_warp(X, sigma=0.2, knot=4):
    """
    Apply time warping to the data using cubic splines.

    Parameters:
    X (numpy.ndarray): Input data array.
    sigma (float): Standard deviation of the time warping.
    knot (int): Number of knots for the spline.

    Returns:
    numpy.ndarray: Time-warped data array.
    """

    X_new = np.zeros_like(X)
    for i in range(X.shape[0]):
        orig_steps = np.arange(X.shape[1])
        random_warp = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
        warp_steps = np.linspace(0, X.shape[1] - 1, num=knot + 2)
        spline = CubicSpline(warp_steps, random_warp)
        new_steps = spline(orig_steps)
        for j in range(X.shape[2]):
            X_new[i, :, j] = np.interp(orig_steps, new_steps, X[i, :, j])
    return X_new

# Permutation

def permute(X, max_segments=5):
    """
    Randomly permute segments of the data.

    Parameters:
    X (numpy.ndarray): Input data array.
    max_segments (int): Maximum number of segments to permute.

    Returns:
    numpy.ndarray: Permuted data array.
    """
    X_new = np.zeros_like(X)
    for i in range(X.shape[0]):
        orig_steps = np.arange(X.shape[1])
        available_points = orig_steps[1:-1]  # Exclude first and last index
        
        # Ensure we do not attempt to take more samples than available
        max_possible_splits = min(len(available_points), max_segments - 1)
        
        if max_possible_splits > 0:
            num_segments = np.random.randint(1, max_possible_splits + 1)
            split_points = np.random.choice(available_points, num_segments - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
            permuted = np.concatenate(np.random.permutation(splits))
        else:
            permuted = orig_steps  # No permutation if not enough points
        
        for j in range(X.shape[2]):
            X_new[i, :, j] = X[i, permuted, j]
    
    return X_new
