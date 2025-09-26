import sys
import argparse
import logging
from os.path import join, dirname
from pathlib import Path

from wearablepermed_utils import __version__

from wearablepermed_utils.core import bin2csv

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# ---- Python API ----

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Bin to CSV Sensor Converter")

    parser.add_argument(
        "--version",
        action="version",
        version=f"uniovi-simur-wearablepermed-utils {__version__}",
    )

    parser.add_argument(
        "-bin-file",
        "--bin-file",        
        dest="bin_file",
        required=True,
        type=str,
        help='Path to the input .BIN file'
    )

    parser.add_argument(
        "-csv-file",
        "--csv-file",            
        dest="csv_file",
        type=str,
        help='Path to the output .CSV file'
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

def create_csv_file(bin_file):
    bin_file_path = Path(bin_file)
    bin_file_dirname = dirname(bin_file)
    csv_file_name = bin_file_path.stem
    
    return join(bin_file_dirname, csv_file_name + '.CSV')

def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.info("Converter starts here")

    # get csv file name
    if args.csv_file == None:
        csv_file = create_csv_file(args.bin_file)
    else:
        csv_file = args.csv_file

    # convert bin file to csv file
    bin2csv(args.bin_file, csv_file)

    _logger.info("Converter ends here")

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
    #     python -m wearablepermed_utils.sensor_bin_to_csv --bin-file /mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/input/PMP1018/PMP1018_W1_C.BIN --csv-file /home/simur/temp/PMP1018_W1_C.csv
    #
    run()
