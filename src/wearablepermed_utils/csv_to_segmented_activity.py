from datetime import datetime
import os
import sys
import argparse
import logging

import numpy as np

from wearablepermed_utils.core import load_segment_wpm_data

from wearablepermed_utils import __version__

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

def parse_time(time_str):
    try:
        return datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        raise argparse.ArgumentTypeError("El formato debe ser HH:MM:SS")
    
def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Process segmented WPM data: load, concatenate, window, and stack from NPZ files",
        epilog="""
        Examples:
        # Basic usage with two files
        python segmented_activity_to_stack.py file1.npz file2.npz --crop-columns 1:7 --window-size 250

        # With step size and output file
        python segmented_activity_to_stack.py file1.npz file2.npz --crop-columns 1:7 --window-size 250 --step-size 125 --output result.npz

        # Using specific columns
        python segmented_activity_to_stack.py file1.npz file2.npz --crop-columns 1,2,3,4,5,6 --window-size 250

        # Process WPM data from examples
        python segmented_activity_to_stack.py ../examples/data/Segmented_WPM_Data/datos_segmentados_PMP1020_W1_PI.npz ../examples/data/Segmented_WPM_Data/datos_segmentados_PMP1020_W1_M.npz --crop-columns 1:7 --window-size 250 --output combined_data.npz -v
        """        
    )   

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
        help='Path to CSV file with MATRIX data'
    )
    
    parser.add_argument(
        "-excel-activity-log",
        "--excel-activity-log",   
        dest="excel_activity_log",
        required=True,
        type=str,
        help='Path to Excel file with activity log'
    )
    
    parser.add_argument(
        "-body-segment",
        "--body-segment",           
        dest="body_segment",
        required=True,
        type=str,
        choices=['PI', 'M', 'C'],
        help='Body segment where the IMU is placed'
    )
    
    parser.add_argument(
        "-plot",
        "--plot",
        dest='plot',
        action='store_true',
        default=True,
        help='Show plots of segmented data (default: True)'
    )
    
    parser.add_argument(
        "-no-plot",
        "--no-plot",        
        dest='no_plot',
        action='store_false',
        help='Do not show plots of segmented data'
    )
    
    parser.add_argument(
        "-output",
        "--output",
        dest="output",
        default=True,
        type=str,
        help='Output file name (without extension) to save segmented data'
    )
    
    parser.add_argument(
        "-sample-init",
        "--sample-init",        
        dest='sample_init',
        type=int,
        help='Sample index for "CAMINAR - USUAL SPEED"'
    )
    
    parser.add_argument(
        "-start-time",
        "--start-time",        
        dest='start_time',
        type=parse_time,
        help='Start time for "CAMINAR - USUAL SPEED"'
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

    _logger.info("csv to segmentation starts here")

    try:        
        # Execute main function
        load_segment_wpm_data(
            csv_file=args.csv_file,
            excel_activity_log=args.excel_activity_log,
            body_segment=args.body_segment,
            plot_data=args.plot,
            out_file=args.output,
            sample_init=args.sample_init,
            start_time=args.start_time
        )
        
        _logger.warning("Processing completed successfully!")
    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _logger.error("✗ Error during processing: {e}", file=sys.stderr)

        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        _logger.info("csv to segmentation ends here")

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
