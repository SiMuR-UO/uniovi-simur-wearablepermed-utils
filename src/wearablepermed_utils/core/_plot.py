import numpy as np                                                                                      #*
import pandas as pd   

def load_MATRIX_data_by_index(csv_file, axes_indices = [1, 2, 3]):
    """Load and process IMU data from a CSV file based on specific indices for axes of the IMU.
    
    Args:
        csv_file (str): Path to the CSV file.
        axes_indices (np.array): Array of indices to select specific IMU axes.
    
    Returns:
        np.array: Array of timestamps and IMU data with the appropriate transformations.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract relevant IMU data (accelerometer and gyroscope)
    imu_data = df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].to_numpy()

    # Apply index adjustments for axes selection
    #It is necessary to do this before sign correction, the order is RELEVANT
    adjusted_indices = np.concatenate([np.abs(axes_indices)-1, np.abs(axes_indices)+2])
    imu_data = imu_data[:, adjusted_indices]
 
    # Apply axis signs based on input indices
    axis_signs = np.concatenate([np.sign(axes_indices), np.sign(axes_indices)])
    imu_data = imu_data * axis_signs

    # Extract timestamps
    timestamps = df['dateTime'].to_numpy().reshape(-1, 1)
 
    #Get remaining fields: temperatures and PPG data
    temp_ppg_data = df[['bodySurface_temp','ambient_temp','hr_raw','hr']].to_numpy()    
    combined_data = np.hstack([timestamps, imu_data, temp_ppg_data])

    return combined_data

def load_WPM_data(csv_file, segment):
    """Load IMU data based on the segment of the body being analyzed (e.g., Wrist, Thigh, Hip).
    
    Args:
        csv_file (str): Path to the CSV file.
        segment (str): Segment of the body (e.g., 'Wrist', 'Thigh').
    
    Returns:
        np.array: Processed IMU data for the specified body segment.
    """
    if segment == "Wrist":
        return load_MATRIX_data_by_index(csv_file, np.array([-1, 3, -2]))
    elif segment == "Thigh":
        return load_MATRIX_data_by_index(csv_file, np.array([3, -1, 2]))
    elif segment == "Hip":
        return load_MATRIX_data_by_index(csv_file, np.array([-1, -3, -2]))

