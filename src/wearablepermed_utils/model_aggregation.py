import os
import sys
import argparse
import logging
from os import walk, path
from pathlib import Path

import numpy as np

from wearablepermed_utils import __version__

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

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
        "-dataset-folder",
        "--dataset-folder",
        dest="dataset_folder",
        required=True,
        help="Select dataset root folder."
    )
    parser.add_argument(
        "-output-folder",
        "--output-folder",
        dest="output_folder",
        required=True,
        help="Select output folder."
    )
    parser.add_argument(
        "-case-id",
        "--case-id",
        dest="case_id",
        required=True,
        help="Select Case Id"
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

def combine_datasets(dataset_folder, output_folder, case_id):
    dataset = []
    dataset_label = []
    dataset_metadata = []
    dataset_feature = []
    dataset_feature_label = []
    dataset_feature_metadata = []

    for dataset_folder_path, participant_ids, filenames in walk(dataset_folder):
        participant_ids.sort()
        filenames.sort()
    
        # Only process one level of subfolders
        if dataset_folder_path == dataset_folder:
            # Execute the pipeline for each participant
            for participant_id in participant_ids:
                # set participant folder
                participant_folder = Path(path.join(dataset_folder, participant_id))

                participant_files = [
                    f for f in os.listdir(participant_folder) 
                    if os.path.isfile(os.path.join(participant_folder, f)) and
                    f.endswith(".npz") and "_all" in f
                ]

                # aggregate datasets
                for participant_file in participant_files:
                    # aggregate not feature datasets: wrist and thing 
                    if "features" not in participant_file:
                        participant_sensor_file = os.path.join(participant_folder, participant_file)
                        participant_sensor_dataset = np.load(participant_sensor_file)
                        
                        dataset.append(participant_sensor_dataset["WINDOW_CONCATENATED_DATA"])
                        dataset_label.append(participant_sensor_dataset["WINDOW_ALL_LABELS"])
                        dataset_metadata.append(participant_sensor_dataset["WINDOW_ALL_METADATA"])
                        
                    # aggregate feature datasets: wrist and thing
                    if "features" in participant_file:
                        participant_sensor_feature_file = os.path.join(participant_folder, participant_file)
                        participant_sensor_feature_dataset = np.load(participant_sensor_feature_file)
                        
                        dataset_feature.append(participant_sensor_feature_dataset["WINDOW_CONCATENATED_DATA"])
                        dataset_feature_label.append(participant_sensor_feature_dataset["WINDOW_ALL_LABELS"])
                        dataset_feature_metadata.append(participant_sensor_feature_dataset["WINDOW_ALL_METADATA"])

            case_id_folder = Path(path.join(output_folder, case_id))
            case_id_folder.mkdir(parents=True, exist_ok=True)

            if len(dataset) > 0:
                dataset = np.concatenate(dataset, axis=1)
                dataset_label = np.concatenate(dataset_label, axis=0)
                dataset_metadata = np.concatenate(dataset_metadata, axis=0)
            
                dataset_all_file = os.path.join(case_id_folder, "data_all.npz")

                np.savez(dataset_all_file, dataset, dataset_label, dataset_metadata)
            
            if len(dataset_feature) > 0:
                dataset_feature = np.concatenate(dataset_feature, axis=1)
                dataset_feature_label = np.concatenate(dataset_feature_label, axis=0)
                dataset_feature_metadata = np.concatenate(dataset_feature_metadata, axis=0)
                        
                dataset_feature_all_file = os.path.join(case_id_folder, "data_feature_all.npz")

                np.savez(dataset_feature_all_file, dataset_feature, dataset_feature_label, dataset_feature_metadata)   

def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.info("Agregator starts here")

    # Participant datasets agregation
    combine_datasets(args.dataset_folder, args.output_folder, args.case_id)

    _logger.info("Agregator end here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
