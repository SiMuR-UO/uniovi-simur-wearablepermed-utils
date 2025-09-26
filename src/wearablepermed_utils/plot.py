import os
import sys
import argparse
import logging

import matplotlib
import matplotlib.pyplot as plt

from wearablepermed_utils import __version__

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

from wearablepermed_utils.core import load_WPM_data

# ---- Python API ----

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version=f"uniovi-simur-wearablepermed-utils {__version__}",
    )

    parser.add_argument(
        "-csv-file",
        "--csv-file",            
        dest="csv_file",
        required=True,
        type=str,
        help='Path to the input .CSV file'
    )

    parser.add_argument(
        "-body-segment",
        "--body-segment",            
        dest="body_segment",
        choices=['Thigh', 'Wrist', 'Hip'],
        required=True,
        help='Segment to be plot: "Thigh", "Wrist", "Hip"'
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

    _logger.info("plot starts here")

    # get sensor data to be ploted
    sensor_data = load_WPM_data(args.csv_file, args.body_segment)

    # plot the data
    plt.figure()
    matplotlib.use('TkAgg') 
    plt.plot(sensor_data[:,1:4])

    filename = os.path.basename(args.csv_file)

    plt.title(filename)
    plt.xlabel('Sample [-]')
    plt.ylabel('Accelerometer data [g]')
    plt.grid(True)
    plt.show()

    _logger.info("Plot ends here")

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
