import sys
import argparse
import logging
import os
import shutil

from wearablepermed_utils import __version__

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# ---- Python API ----
    
def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Clean padataset folder",
        epilog="""
        Examples:
        # Basic usage:
        clean_dataset_folder \ 
            --dataset-folder /mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input
        """        
    )   

    parser.add_argument(
        "--version",
        action="version",
        version=f"uniovi-simur-wearablepermed-utils {__version__}",
    )

    parser.add_argument(
        "-dataset-folder",
        "--dataset-folder",            
        dest="dataset_folder",
        required=True,
        help='Path to dataset folder'
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )

    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
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

def main(args): 
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.info("Clean Dataset folder starts here")


    for subfolder in sorted(os.listdir(args.dataset_folder)):
        folder_path = os.path.join(args.dataset_folder, subfolder)

        if not os.path.isdir(folder_path):
            continue  # skip files if any

        # Remove all .npz files in the current subfolder
        for file in os.listdir(folder_path):
            if file.endswith(".npz"):
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)
                _logger.info("Deleted file " + file_path)

        # Remove 'images_activities' folder if it exists
        images_folder = os.path.join(folder_path, "images_activities")
        if os.path.exists(images_folder) and os.path.isdir(images_folder):
            shutil.rmtree(images_folder)
            _logger.info("Deleted folder " + images_folder)

        _logger.info("Clean Dataset end here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m wearablepermed_utils.plot --csv-file /mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1018/PMP1018_W1_Acelerometria_M.CSV --segment M
    #
    run()