#!/usr/bin/env python3
"""
Script para extraer features desde un archivo NPZ de stack de datos enventanados.
Utiliza la función extract_features_from_stack del módulo feature_extraction.
"""

import os
import sys
import argparse
import logging

import numpy as np

from wearablepermed_utils import __version__

__author__ = "Miguel Angel Salinas Gancedo<uo34525@uniovi.es>, Alejandro Castellanos Alonso<uo265351@uniovi.es>, Antonio Miguel López Rodriguez<amlopez@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

from wearablepermed_utils.core import extract_features_from_stack

# ---- Python API ----

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """

    parser = argparse.ArgumentParser(
        description="Extract features from a stacked NPZ file containing windowed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic usage with default parameters
            python stack_to_features.py data_stack.npz --output features.npz

            # Specify number of IMUs
            python stack_to_features.py data_stack.npz --n-imus 1 --output features.npz

            # With verbose output
            python stack_to_features.py data_stack.npz --n-imus 2 --output features.npz --verbose

            # Real example with project data
            python stack_to_features.py ../examples/data/stacks/data_tot_PMP1020_1051.npz --output features_extracted.npz --verbose
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"uniovi-simur-wearablepermed-utils {__version__}",
    )

    parser.add_argument(
        '-stack-file',
        '--stack-file',
        dest='stack_file',
        required=True,
        type=str,
        help='Path to the NPZ file containing stacked windowed data'
    )
    
    # parser.add_argument(
    #     '-n-imus',
    #     '--n-imus',
    #     dest='n_imus',
    #     type=int,
    #     default=1,
    #     help='Number of IMUs in the stack data (default: 1)'
    # )
    
    parser.add_argument(
        '-output', 
        '--output', 
        dest='output',
        type=str,
        required=True,
        help='Output NPZ file path to save extracted features'
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
            
    try:
        # Extract features from stack
        # result = extract_features_from_stack(args.stack_file, n_imus=args.n_imus)
        features, labels, metadata = extract_features_from_stack(args.stack_file)
        
        # if args.loglevel in [logging.INFO, logging.DEBUG]:
        #     print("✓ Feature extraction completed successfully!")
        #     print(f"✓ Features shape: {result['features'].shape}")
        #     print(f"✓ Number of windows: {result['num_windows']}")
        #     print(f"✓ Data shape: {result['data_shape']}")
        #     print(f"✓ Unique labels: {len(result['unique_labels'])}")
        #     print(f"✓ Labels: {list(result['unique_labels'])}")
        #     print()
            
        #     # Show label distribution
        #     unique_labels, counts = np.unique(result['labels'], return_counts=True)
        #     print("Label distribution:")
        #     for label, count in zip(unique_labels, counts):
        #         print(f"  {label}: {count} windows")
        #     print()
        
        # Save results to NPZ file
        np.savez(
            args.output,
            features,
            labels,
            metadata
        )
        
        if args.loglevel in [logging.INFO, logging.DEBUG]:
            print(f"✓ Results saved to: {args.output}")
            file_size = os.path.getsize(args.output)
            print(f"✓ Output file size: {file_size} bytes")
        else:
            print(f"Features extracted and saved to: {args.output}")
            
    except Exception as e:
        print(f"Error during feature extraction: {e}", file=sys.stderr)
        if args.loglevel == logging.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)

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