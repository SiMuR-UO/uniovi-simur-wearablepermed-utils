import os
import sys
import argparse
import logging

import numpy as np

from wearablepermed_utils.core import load_concat_window_stack

from wearablepermed_utils import __version__

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# ---- Python API ----

_DEF_WINDOW_OVERLAPPING_PERCENT = None

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
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
        '-npz-files', 
        '--npz-files',
        dest="npz_files",
        help='Paths to NPZ files to process',
        nargs='+',        
    )
    
    parser.add_argument(
        '-crop-columns', 
        '--crop-columns', 
        dest="crop_columns",
        help='Columns to select from arrays. Format: "start:end" or "col1,col2,col3". Default: "1:7"',        
        type=parse_crop_columns,
        default=slice(1, 7),
        required=True
    )
    
    parser.add_argument(
        '-window-size', 
        '--window-size',
        dest="window_size",
        help='Window size in number of samples',        
        type=int,
        required=True,
    )
    
    parser.add_argument(
        "-window-overlapping-percent",
        "--window-overlapping-percent",
        type=int,
        default=_DEF_WINDOW_OVERLAPPING_PERCENT,
        dest="window_overlapping_percent", 
        help="Window Overlapping percent.")  
    
    parser.add_argument(
        "-include-not-estructure-data",
        "--include-not-estructure-data",
        dest="include_not_estructure_data",
        action='store_true',
        help="Include estructure data.")     
        
    parser.add_argument(
        '-output', 
        '--output',  
        help='Output file name to save results (.npz format)',               
        type=str
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

def parse_crop_columns(crop_str):
    """Parse crop columns argument from string to slice or list."""
    if crop_str is None:
        return slice(None)
    
    try:
        # Try to parse as slice notation (e.g., "1:7")
        if ':' in crop_str:
            parts = crop_str.split(':')
            if len(parts) == 2:
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if parts[1] else None
                return slice(start, end)
            elif len(parts) == 3:
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if parts[1] else None
                step = int(parts[2]) if parts[2] else None
                return slice(start, end, step)
        else:
            # Try to parse as comma-separated list (e.g., "1,2,3,4,5,6")
            return [int(x.strip()) for x in crop_str.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid crop columns format: {crop_str}")

def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.info("Segmentation starts here")
     
    try:
        # Execute the main function
        stacked_data, labels = load_concat_window_stack(
            args,
            npz_file_paths=args.npz_files,
            crop_columns=args.crop_columns,
            window_size_samples=args.window_size,
            window_overlapping_percent=args.window_overlapping_percent,
            save_file_name=args.output
        )
        
        _logger.debug("✓ Processing completed successfully!")
        _logger.debug("✓ Stacked data shape: {stacked_data.shape}")
        _logger.debug("✓ Number of labels: {len(labels)}")
        _logger.debug("✓ Unique activities: {len(np.unique(labels))}")

        if args.loglevel == logging.DEBUG:            
            # Show label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)

            _logger.debug("Label distribution:")
            for label, count in zip(unique_labels, counts):
                _logger.debug("{label}: {count} windows")
        
        if args.output:
            _logger.debug("✓ Results saved to: {args.output}")
        else:
            _logger.debug("ℹ Results not saved (use --output to save)")      
    except Exception as e:
        _logger.error("✗ Error during processing: {e}", file=sys.stderr)

        if args.loglevel == logging.DEBUG: 
            import traceback
            traceback.print_exc()

        sys.exit(1)
    finally:
        _logger.info("Segmentation end here")

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
    #     python -m wearablepermed_utils.segmented_activity_to_stack --npz-files file1.npz file2.npz --crop-columns 1:7 --window-size 250
    #
    run()
