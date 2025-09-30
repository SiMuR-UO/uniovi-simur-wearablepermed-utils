import os
import sys
import argparse
import logging
from enum import Enum

import numpy as np

from wearablepermed_utils import __version__

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# ---- Python API ----

WINDOW_CONCATENATED_DATA = "arr_0"
WINDOW_ALL_LABELS = "arr_1"
WINDOW_ALL_METADATA = "arr_2"

class ML_Model(Enum):
    ESANN = 'ESANN'
    CAPTURE24 = 'CAPTURE24'
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'

class ML_Sensor(Enum):
    PI = 'thigh'
    M = 'wrist'
    C = 'hip'

def parse_ml_model(value):
    try:
        """Parse a comma-separated list of CML Models lor values into a list of ML_Sensor enums."""
        values = [v.strip() for v in value.split(',') if v.strip()]
        result = []
        invalid = []
        for v in values:
            try:
                result.append(ML_Model(v))
            except ValueError:
                invalid.append(v)
        if invalid:
            valid = ', '.join(c.value for c in ML_Model)
            raise argparse.ArgumentTypeError(
                f"Invalid color(s): {', '.join(invalid)}. "
                f"Choose from: {valid}"
            )
        return result
    except ValueError:
        valid = ', '.join(ml_model.value for ml_model in ML_Model)
        raise argparse.ArgumentTypeError(f"Invalid ML Model '{value}'. Choose from: {valid}")
    
def parse_ml_sensor(value):
    try:
        """Parse a comma-separated list of CML Models lor values into a list of ML_Sensor enums."""
        values = [v.strip() for v in value.split(',') if v.strip()]
        result = []
        invalid = []
        for v in values:
            try:
                result.append(ML_Sensor(v))
            except ValueError:
                invalid.append(v)
        if invalid:
            valid = ', '.join(c.value for c in ML_Sensor)
            raise argparse.ArgumentTypeError(
                f"Invalid color(s): {', '.join(invalid)}. "
                f"Choose from: {valid}"
            )
        return result
    except ValueError:
        valid = ', '.join(ml_model.value for ml_model in ML_Sensor)
        raise argparse.ArgumentTypeError(f"Invalid ML Model '{value}'. Choose from: {valid}")
    
def parse_args(args):
    """Parse command line parameters

    Args:hip
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Machine Learning Model Trainer")
   
    parser.add_argument(
        "-ml-models",
        "--ml-models",
        type=parse_ml_model,
        nargs='+',
        dest="ml_models",        
        required=True,
        help=f"Available ML models: {[c.value for c in ML_Model]}."
    )
    parser.add_argument(
        "-participant",
        "--participant",
        dest="participant",
        required=True,
        type=str,
        help="Participant for which we aggregate the window .npz files."
    ) 
    parser.add_argument(
        "-ml-sensors",
        "--ml-sensors",
        type=parse_ml_sensor,
        nargs='+',
        dest="ml_sensors",        
        required=True,
        help=f"Available ML sensors: {[c.value for c in ML_Sensor]}."
    )
    parser.add_argument(
        "-dataset-folder",
        "--dataset-folder",
        dest="dataset_folder",
        required=True,
        help="Choose the dataset root folder."
    )
    parser.add_argument(
        "-case-id-folder",
        "--case-id-folder",
        dest="case_id_folder",
        required=True,
        help="Choose the case id root folder."
    )      
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO.",
        action="store_const",
        const=logging.INFO,
    )
  
    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

def convolution_model_selected(models):
    for model in models:
        if model.value in [ML_Model.CAPTURE24.value, ML_Model.ESANN.value]:
            return True
        
    return False

def feature_model_selected(models):
    for model in models:
        if model.value in [ML_Model.RANDOM_FOREST.value, ML_Model.XGBOOST.value]:
            return True
        
    return False

def combine_participant_dataset(dataset_folder, participant, models, sensors):
    participant_folder = os.path.join(dataset_folder, participant)
    participant_files = [
        f for f in os.listdir(participant_folder) 
        if os.path.isfile(os.path.join(participant_folder, f)) and
        ".npz" in f and
        any(sensor in f for sensor in ['_' + item for item in [sensor.name for sensor in sensors]])
    ]

    participant_files = sorted(participant_files)

    participant_dataset = []
    participant_label_dataset = []
    participant_metadata_dataset = []

    participant_feature_dataset = []
    participant_feature_label_dataset = []
    participant_feature_metadata_dataset = []
    
    # aggregate datasets
    for participant_file in participant_files:
        # aggregate not feature datasets: wrist and thing 
        if "features" not in participant_file and convolution_model_selected(models) and "tot" in participant_file:
            participant_sensor_file = os.path.join(participant_folder, participant_file)
            participant_sensor_dataset = np.load(participant_sensor_file)
            
            participant_dataset.append(participant_sensor_dataset[WINDOW_CONCATENATED_DATA])
            participant_label_dataset.append(participant_sensor_dataset[WINDOW_ALL_LABELS])
            participant_metadata_dataset.append(participant_sensor_dataset[WINDOW_ALL_METADATA])
            
         # aggregate feature datasets: wrist and thigh
        if "features" in participant_file and "mets" not in participant_file and feature_model_selected(models) and "tot" in participant_file:
            participant_sensor_feature_file = os.path.join(participant_folder, participant_file)
            participant_sensor_feature_dataset = np.load(participant_sensor_feature_file)
            
            participant_feature_dataset.append(participant_sensor_feature_dataset[WINDOW_CONCATENATED_DATA])
            participant_feature_label_dataset.append(participant_sensor_feature_dataset[WINDOW_ALL_LABELS])
            participant_feature_metadata_dataset.append(participant_sensor_feature_dataset[WINDOW_ALL_METADATA])

    if len(participant_dataset) > 0:
        participant_dataset = np.concatenate(participant_dataset, axis=1)
        participant_label_dataset = participant_label_dataset[:participant_dataset.shape[0]][0]
        participant_metadata_dataset = participant_metadata_dataset[:participant_dataset.shape[0]][0]

        participant_sensor_all_file = os.path.join(participant_folder, 'data_' + participant + '_all.npz')
        np.savez(participant_sensor_all_file, participant_dataset, participant_label_dataset, participant_metadata_dataset)
    
    if len(participant_feature_dataset) > 0:
        _logger.info(participant)

        participant_feature_dataset = np.concatenate(participant_feature_dataset, axis=1)
        participant_feature_label_dataset = participant_feature_label_dataset[:participant_feature_dataset.shape[0]][0]
        participant_feature_metadata_dataset = participant_feature_metadata_dataset[:participant_feature_dataset.shape[0]][0]
                
        participant_sensor_feature_all_file = os.path.join(participant_folder, 'data_' + participant + "_features_all.npz")
        np.savez(participant_sensor_feature_all_file, participant_feature_dataset, participant_feature_label_dataset, participant_feature_metadata_dataset) 
                
                             
def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.info("Agregator starts here")

    # Participant datasets agregation
    if len(args.ml_sensors[0]) > 0:
        combine_participant_dataset(args.dataset_folder, args.participant, args.ml_models[0], args.ml_sensors[0])

    _logger.info("Agregator end here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
