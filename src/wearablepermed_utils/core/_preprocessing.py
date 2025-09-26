import numpy as np
from scipy.interpolate import CubicSpline

def NORM(senhal):
    """Calculate the Euclidean norm (L2 norm) of the input signal along axis 1.
    
    Args:
        senhal (np.array): A 2D array where each row represents a vector (e.g., accelerometer data).
        Each column is a signal and each row is a time sample
    
    Returns:
        np.array: Euclidean norm of the input signal, calculated along axis 1.
    """
    # Compute the Euclidean norm for each row (vector) in the input signal
    return np.linalg.norm(np.atleast_2d(senhal), axis=1)

def ENMO(senhal, G=1):
    """Calculate the Euclidean Norm Minus One (ENMO) of the input signal.
    
    This metric is commonly used in accelerometer data to remove the gravitational component (assumed to be 1).
    
    Args:
        senhal (np.array): A 2D array where each row represents a vector.
        Each column is a signal and each row is a time sample
        G: value to substract (default is 1, G units), optinal G = 9.81 (m/s^2, terrestrial gravity)
   
    Returns:
        np.array: The ENMO value, which is the Euclidean norm minus G for each row.
    """
    # Subtract 1 from the Euclidean norm of the signal to remove the gravitational constant
    return NORM(senhal) - G

def MAD(senhal):
    """Calculate the Mean Absolute Deviation (MAD) of the signal's magnitude.
    
    MAD measures the average deviation of the signal from its mean, providing a metric of variability.
    
    Args:
        senhal (np.array): A 2D array where each row represents a vector, or a 1D array representing a single vector.
        Each column is a signal and each row is a time sample
   
    Returns:
        np.array: The mean absolute deviation of the signal's magnitude for each row, or a single value if the input is 1D.
    """
    # Ensure the input is a numpy array
    
    magnitude = NORM(senhal)
    mean_values = np.mean(magnitude)
    
    # Calculate the absolute deviations from the mean
    deviations = np.abs(magnitude - mean_values)
    
    # Calculate the mean of the absolute deviations (MAD)
    mad = np.mean(deviations)
    
    return mad

def clip_data(data, index=[0, 1, 2], clip_value=8):
    """Clips the values in the specified rows of the data matrix to a given clip value.

    Parameters:
    data (numpy.ndarray): A 2D array where each column represents a signal and each row represents a temporal value.
    index (list of int, optional): The columns of the matrix to apply the clipping. Default is [0, 1, 2].
    clip_value (int, optional): The value to clip the data to. Default is 8.

    Returns:
    None: The function modifies the input data matrix in place.

    Notes:
    - Values greater than clip_value are set to clip_value.
    - Values less than -clip_value are set to -clip_value.
    """
    for k in index:
        data[:, k] = np.clip(data[:, k], -clip_value, clip_value)
                
def time_interp(data, Tm, t_index=0):
    """
    Interpolates the given data matrix over a new time vector.

    Parameters:
    data (numpy.ndarray): A 2D array where columns represent different signals and rows represent time points.
    Tm (float): The time interval for the new time vector.
    t_index (int): The column index of the time values in the data matrix.

    Returns:
    numpy.ndarray: A new matrix with the first column as the new time vector and the subsequent columns as the interpolated signals.
    """
    new_t = np.arange(data[0, t_index], data[-1, t_index] + Tm, Tm)
    interpolated_signals = [new_t.reshape((len(new_t), 1))]

    for idx in range(data.shape[1]):
        if idx != t_index:
            new_d = np.interp(new_t, data[:, t_index], data[:, idx])
            interpolated_signals.append(new_d.reshape((len(new_t), 1)))

    interpolated = np.hstack(interpolated_signals)
    return interpolated
