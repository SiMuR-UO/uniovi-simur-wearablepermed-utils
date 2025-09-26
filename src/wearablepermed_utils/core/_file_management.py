import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl                                        # Librería para leer datos desde una Hoja Excel
from datetime import time, timedelta, date, datetime
import matplotlib.pyplot as plt

import openpyxl
from datetime import time, timedelta, date

#__all__ = ['load_WPM_IMU_data', 'segment_data_by_dates']

def read_time_from_excel(file_path, sheet_name, cell_reference):
    """
    Reads a specific cell from an Excel file that contains a time in the format 'hh:mm:ss' 
    or 'h:mm:ss' and converts it into a Python 'time' object.

    :param file_path: Full path to the Excel file.
    :param sheet_name: Name of the sheet where the cell is located.
    :param cell_reference: Cell reference that contains the time (e.g., 'A1').
    :return: A 'time' object with the converted time, or None if the format is invalid.
    """
    # Load the Excel file
    workbook = openpyxl.load_workbook(file_path, data_only=True)

    # Select the worksheet
    sheet = workbook[sheet_name]

    # Read the cell value
    cell_value = sheet[cell_reference].value

    # If the value is a timedelta (accumulated hours)
    if isinstance(cell_value, timedelta):
        total_seconds = cell_value.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return time(hours, minutes, seconds)

    # If the value is already a time object
    elif isinstance(cell_value, time):
        return cell_value

    # If the value is a string in the format 'h:mm:ss' or 'hh:mm:ss'
    elif isinstance(cell_value, str):
        try:
            # Split the time string into its components
            time_parts = cell_value.split(":")
            
            if len(time_parts) == 3:
                hours, minutes, seconds = map(int, time_parts)
                return time(hours, minutes, seconds)
            else:
                print(f"Invalid format in cell {cell_reference} (expected h:mm:ss).")
                return None
        except ValueError:
            print(f"Invalid time value in cell {cell_reference}.")
            return None

    # If the value is a number (Excel can store time as fractions of a day)
    elif isinstance(cell_value, (int, float)):
        try:
            total_seconds = cell_value * 24 * 3600  # Convert the value to seconds
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            return time(hours, minutes, seconds)
        except ValueError:
            print(f"Numeric value in cell {cell_reference} couldn't be converted.")
            return None

    # If the value is not a valid type
    else:
        print(f"Invalid type in cell {cell_reference}. Expected timedelta, string, or number.")
        return None

def read_date_from_excel(file_path, sheet_name, cell_reference):
    """
    Reads a specific cell from an Excel file that contains a date in the format 'day/month/year'
    and converts it into a Python 'date' object.

    :param file_path: Full path to the Excel file.
    :param sheet_name: Name of the sheet where the cell is located.
    :param cell_reference: Cell reference that contains the date (e.g., 'A1').
    :return: A 'date' object with the converted date, or None if the format is invalid.
    """
    # Load the Excel file
    workbook = openpyxl.load_workbook(file_path, data_only=True)

    # Select the worksheet
    sheet = workbook[sheet_name]

    # Read the cell value
    cell_value = sheet[cell_reference].value

    # If the value is already a date object
    if isinstance(cell_value, date):
        return cell_value

    # If the value is a string in the format 'day/month/year'
    elif isinstance(cell_value, str):
        try:
            date_parts = cell_value.split("/")
            if len(date_parts) == 3:
                day, month, year = map(int, date_parts)
                return date(year, month, day)
            else:
                print(f"Invalid date format in cell {cell_reference} (expected day/month/year).")
                return None
        except ValueError:
            print(f"Invalid date value in cell {cell_reference}.")
            return None

    # If the value is a number (Excel stores dates as days since January 1, 1900)
    elif isinstance(cell_value, (int, float)):
        try:
            excel_start_date = date(1900, 1, 1).toordinal() + int(cell_value) - 2  # Excel adjustment
            return date.fromordinal(excel_start_date)
        except ValueError:
            print(f"Numeric value in cell {cell_reference} couldn't be converted to a date.")
            return None

    # If the value is not a valid type
    else:
        print(f"Invalid type in cell {cell_reference}. Expected date, string, or number.")
        return None

def extract_WPM_info_from_excel(file_path):
    """
    Extracts time and date information from specific cells in an Excel file related to 
    a WPM study, and prints it in a formatted manner.
    
    :param file_path: Full path to the Excel file.
    :return: A dictionary containing extracted times and dates.
    """
    time_data = {}

    # Define cell locations and labels
    cell_definitions = [
        ("E37", "Hora de inicio de acelerómetro muslo (hh:mm:ss) - Hora de ordenador"),
        ("E38", "Hora de inicio de acelerómetro cadera (hh:mm:ss) - Hora de ordenador"),
        ("E39", "Hora de inicio de acelerómetro muñeca (hh:mm:ss) - Hora de ordenador"),
        ("E13", "Fecha día 1"),
        ("D60", "FASE REPOSO CON K5 - Hora de inicio"),
        ("D61", "FASE REPOSO CON K5 - Hora de fin"),
        ("D72", "TAPIZ RODANTE - Hora de inicio"),
        ("D73", "TAPIZ RODANTE - Hora de fin"),
        ("D81", "SIT TO STAND 30 s - Hora de inicio"),
        ("D82", "SIT TO STAND 30 s - Hora de fin"),
        ("D90", "INCREMENTAL CICLOERGOMETRO - Hora de inicio REPOSO"),
        ("D91", "INCREMENTAL CICLOERGOMETRO - Hora de inicio CALENTAMIENTO"),
        ("D92", "INCREMENTAL CICLOERGOMETRO - Hora de inicio INCREMENTAL"),
        ("D93", "INCREMENTAL CICLOERGOMETRO - Hora de fin"),
        ("D104", "ACTIVIDAD NO ESTRUCTURADA - Hora de inicio"),
        ("D115", "ACTIVIDAD NO ESTRUCTURADA - Hora de fin"),
        ("E112", "Fecha día 7"),
        ("D144", "YOGA - Hora de inicio"),
        ("D145", "YOGA - Hora de fin"),
        ("D153", "SENTADO VIENDO LA TV - Hora de inicio"),
        ("D154", "SENTADO VIENDO LA TV - Hora de fin"),
        ("D162", "SENTADO LEYENDO - Hora de inicio"),
        ("D163", "SENTADO LEYENDO - Hora de fin"),
        ("D172", "SENTADO USANDO PC - Hora de inicio"),
        ("D173", "SENTADO USANDO PC - Hora de fin"),
        ("D181", "DE PIE USANDO PC - Hora de inicio"),
        ("D182", "DE PIE USANDO PC - Hora de fin"),
        ("D190", "DE PIE DOBLANDO TOALLAS - Hora de inicio"),
        ("D191", "DE PIE DOBLANDO TOALLAS - Hora de fin"),
        ("D199", "DE PIE MOVIENDO LIBROS - Hora de inicio"),
        ("D200", "DE PIE MOVIENDO LIBROS - Hora de fin"),
        ("D208", "DE PIE BARRIENDO - Hora de inicio"),
        ("D209", "DE PIE BARRIENDO - Hora de fin"),
        ("D219", "CAMINAR USUAL SPEED - Hora de inicio"),
        ("D220", "CAMINAR USUAL SPEED - Hora de fin"),
        ("D228", "CAMINAR CON MÓVIL O LIBRO - Hora de inicio"),
        ("D229", "CAMINAR CON MÓVIL O LIBRO - Hora de fin"),
        ("D237", "CAMINAR CON LA COMPRA - Hora de inicio"),
        ("D238", "CAMINAR CON LA COMPRA - Hora de fin"),
        ("D246", "CAMINAR ZIGZAG - Hora de inicio"),
        ("D247", "CAMINAR ZIGZAG - Hora de fin"),
        ("D255", "TROTAR - Hora de inicio"),
        ("D256", "TROTAR - Hora de fin"),
        ("D264", "SUBIR Y BAJAR ESCALERAS - Hora de inicio"),
        ("D265", "SUBIR Y BAJAR ESCALERAS - Hora de fin")
    ]

    # Loop through each defined cell to extract and store data
    for cell, label in cell_definitions:
        if "fecha" in label.lower():
            result = read_date_from_excel(file_path, 'Hoja1', cell)
        else:
            result = read_time_from_excel(file_path, 'Hoja1', cell)

        time_data[label] = result
        
        """
        if result:
            if isinstance(result, time):
                print(f"{label}: {result.strftime('%H:%M:%S')}")
            elif isinstance(result, date):
                print(f"{label}: {result.strftime('%Y/%m/%d')}")
        """

    return time_data

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
    if segment == "M":
        return load_MATRIX_data_by_index(csv_file, np.array([-1, 3, -2]))
    elif segment == "PI":
        return load_MATRIX_data_by_index(csv_file, np.array([3, -1, 2]))
    elif segment == "C":
        return load_MATRIX_data_by_index(csv_file, np.array([-1, -3, -2]))

def calculate_accelerometer_drift(WPM_data, excel_file_path, body_segment, walk_usual_speed_start_sample=None, start_time_WALKING_USUAL_SPEED=None):
    """
    This function calculates the drift of the MATRIX accelerometer by determining the scaling
    required between the timestamp data from the MATRIX .CSV file and the corresponding Activity Log.

    Parameters:
    -----------
    * WPM_data: np.array containing the data from the MATRIX .CSV file (m samples and 11 features).
    * excel_file_path: Path to the corresponding Activity Log (Excel) in the PMP dataset.
    * body_segment: string, Body segment where the IMU is placed ("Thigh", "Wrist" or "Hip").
    * walk_usual_speed_start_sample: Sample, identified through visual inspection, corresponding to
      the start of the "WALK-USUAL SPEED" activity. Default is None if not specified.

    Returns:
    --------
    * K: float, scaling factor for the MATRIX timestamps.
    """

    # MATRIX power-on/off timestamps from the Excel log
    imu_power_on_date_cell = "E13"  # Cell with the MATRIX power-on date
    imu_power_off_date_cell = "E112"  # Cell with the MATRIX power-off date

    # IMU placed on the thigh
    if body_segment == "PI":
        imu_power_on_time_cell = "E37"  # Cell with the MATRIX power-on time
        imu_power_off_time_cell = "E273"  # Cell with the MATRIX power-off time

    # IMU placed on the hip
    elif body_segment == "C":
        imu_power_on_time_cell = "E38"
        imu_power_off_time_cell = "E274"

    # IMU placed on the wrist
    elif body_segment == "M":
        imu_power_on_time_cell = "E39"
        imu_power_off_time_cell = "E275"

    # *************************** MATRIX EXPERIMENTAL TIMESTAMPS ********************************
    initial_matrix_timestamp_ms = WPM_data[0, 0]  # Read the first MATRIX timestamp
    print(f"Initial MATRIX timestamp: {initial_matrix_timestamp_ms}")

    last_matrix_timestamp_ms = WPM_data[-1, 0]  # Last MATRIX timestamp (in milliseconds)
    print(f"Last MATRIX timestamp recorded: {last_matrix_timestamp_ms}")

    # Calculate the difference between power-on and power-off timestamps
    matrix_timestamp_difference_ms = last_matrix_timestamp_ms - initial_matrix_timestamp_ms
    if not walk_usual_speed_start_sample:
        matrix_power_off_date = read_date_from_excel(excel_file_path, "Hoja1", imu_power_off_date_cell)
        matrix_power_off_time = read_time_from_excel(excel_file_path, "Hoja1", imu_power_off_time_cell)
    else:
        walk_start_matrix_timestamp_ms = WPM_data[walk_usual_speed_start_sample, 0]
        matrix_timestamp_difference_ms = walk_start_matrix_timestamp_ms - initial_matrix_timestamp_ms
        matrix_power_off_date = None
        matrix_power_off_time = None

    print(f"MATRIX timestamp difference [ms]: {matrix_timestamp_difference_ms}")
    # *******************************************************************************************

    # *************************** EXCEL TIMESTAMPS ********************************
    matrix_power_on_date = read_date_from_excel(excel_file_path, "Hoja1", imu_power_on_date_cell)
    matrix_power_on_time = read_time_from_excel(excel_file_path, "Hoja1", imu_power_on_time_cell)
    matrix_power_on_datetime = datetime.combine(matrix_power_on_date, matrix_power_on_time)
    
    excel_power_on_timestamp_sec = matrix_power_on_datetime.timestamp()
    excel_power_on_timestamp_ms = excel_power_on_timestamp_sec * 1000  # Convert to milliseconds
    
    print(f"Excel power-on timestamp (MATRIX): {excel_power_on_timestamp_ms}")

    if (not matrix_power_off_date) and (not walk_usual_speed_start_sample):
        print("Timestamps will not be rescaled due to insufficient data.")
        K = 1
        return K
    elif matrix_power_off_date is None:
        walk_start_date_cell = "E112"
        # walk_start_time_cell = "D219"
        walk_start_date = read_date_from_excel(excel_file_path, "Hoja1", walk_start_date_cell)
        # walk_start_time = read_time_from_excel(excel_file_path, "Hoja1", walk_start_time_cell)
        # We use the .csv file with the sample number and the start time of a known activity
        walk_start_time = start_time_WALKING_USUAL_SPEED
        walk_start_datetime = datetime.combine(walk_start_date, walk_start_time)
        excel_walk_start_timestamp_sec = walk_start_datetime.timestamp()
        excel_walk_start_timestamp_ms = excel_walk_start_timestamp_sec * 1000
        print(f"Excel walk-start timestamp: {excel_walk_start_timestamp_ms}")
        excel_timestamp_difference_ms = excel_walk_start_timestamp_ms - excel_power_on_timestamp_ms
    else:
        matrix_power_off_datetime = datetime.combine(matrix_power_off_date, matrix_power_off_time)
        excel_power_off_timestamp_sec = matrix_power_off_datetime.timestamp()
        excel_power_off_timestamp_ms = excel_power_off_timestamp_sec * 1000
        print(f"Excel power-off timestamp (MATRIX): {excel_power_off_timestamp_ms}")
        excel_timestamp_difference_ms = excel_power_off_timestamp_ms - excel_power_on_timestamp_ms

    if excel_timestamp_difference_ms is not None:
        print(f"Excel timestamp difference [ms]: {excel_timestamp_difference_ms}")
        # Calculate scaling factor
        K = matrix_timestamp_difference_ms / excel_timestamp_difference_ms
        print(f"Timestamp scaling factor: {K}")
        print("Timestamps successfully rescaled.")
    else:
        print("Timestamps will not be rescaled due to insufficient data.")
        K = 1

    return K

def apply_scaling_to_matrix_data(WPM_data, K):
    """
    This function scales the timestamps from the MATRIX data using the scaling factor K calculated
    by the "calculate_accelerometer_drift" function.

    Parameters:
    -----------
    * WPM_data: np.array containing the data from the MATRIX .CSV file (m samples and 11 features).
    * K: float, scaling factor for the MATRIX timestamps.

    Returns:
    --------
    * WPM_data_scaled: np.array with the timestamps scaled.
    """

    original_timestamps = WPM_data[:, 0]  # Extract original MATRIX timestamps (column 0)
    WPM_data_scaled = WPM_data.copy()  # Copy the accelerometer data
    scaled_timestamps = (original_timestamps - original_timestamps[0]) / K + original_timestamps[0]  # Scale timestamps
    WPM_data_scaled[:, 0] = scaled_timestamps  # Update the timestamps in the copied data

    return WPM_data_scaled

def load_scale_WPM_data(csv_file_PMP, segment_body, excel_file_path, calibrate_with_start_WALKING_USUAL_SPEED, start_time_WALKING_USUAL_SPEED):
    """
    This function encapsulates the code to perform load and scaling of WPM data
    Segmentation is not applied in this function.
    
    - Input Parameters:
    --------------------
    * csv_file_PMP: string, path to the ".csv" file containing all data 
      recorded by MATRIX.
    * segment_body: string, body segment where the IMU is placed 
      ("Thigh", "Wrist", or "Hip").
    * excel_file_path: string, path to the corresponding Activity 
      Log of the PMP dataset.
    * calibrate_with_start_WALKING_USUAL_SPEED: int. The sample, visually 
      inspected, that corresponds to the start of the "WALKING-USUAL SPEED" 
      activity. If not specified, its default value is None.
      
    - Return Value:
    --------------------
    Returns WPM data properly scaled and the corresponding dictionary timing from the Excel file.
    """
    
    # ********************************** DATA READING ***************************************
    WPM_data_W1 = load_WPM_data(csv_file_PMP, segment_body)                                                   # Read data: accelerometer placed on a body segment
    dictionary_timing_WPM_PMP = extract_WPM_info_from_excel(excel_file_path)                         # Read timestamps stored in the cells of the activity log

    # ******************************* TIMESTAMP SCALING *************************************
    K = calculate_accelerometer_drift(WPM_data_W1, excel_file_path, segment_body, calibrate_with_start_WALKING_USUAL_SPEED, start_time_WALKING_USUAL_SPEED) # Calculate scaling factor for MATRIX timestamps
    WPM_data_PMP_W1_SCALED = apply_scaling_to_matrix_data(WPM_data_W1, K)                                                              # Apply scaling
    
    return WPM_data_PMP_W1_SCALED, dictionary_timing_WPM_PMP

if __name__ == "__main__":
	print("main empty")
    