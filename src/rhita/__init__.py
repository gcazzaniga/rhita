"""
RHITA (Real-time Hazard Identification and Tracking Algorithm)
=============================================================
RHITA is an algorithm for the detection and tracking of hazards. It requires a configuration file (e.g., `whatevername.ini`)
as its primary input.

Author: Greta Cazzaniga
Last Updated: 14 March 2025
"""

import os
import configparser
import argparse
import logging
import time as t
from datetime import datetime

# Import RHITA functions
import rhita.main as main
import rhita.data_processing as dp

def configure_logging():
    """
    Configure logging for the RHITA application.
    Logs are written to 'app_RHITA.log'.
    """
    if os.path.exists('app_RHITA.log'):
        # Clear the log file if it already exists
        with open('app_RHITA.log', 'w'):
            pass

    logging.basicConfig(
        filename='app_RHITA.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Application RHITA started at {datetime.now()}")

def load_configuration(config_file):
    """
    Load the configuration file using configparser.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        configparser.ConfigParser: Parsed configuration object.
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def run(configfile = None):
    """
    Main function to execute the RHITA algorithm.
    """
    start_time = t.time()

    # Configure logging
    configure_logging()

    if configfile is None:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Run RHITA.")
        parser.add_argument(
            '--configfile',
            type=str,
            required=True,
            help='Path to the configuration file (e.g., whatevername.ini).'
        )
        args = parser.parse_args()

        # Load configuration file
        config = load_configuration(args.configfile)
    else:
        # Load configuration file directly
        config = load_configuration(configfile)
  
    # Input data upload
    ds = dp.import_data(config)

    # Threshold map upload (if available)
    if config['methods_parameters']['threshold1'] in ['map', 'map_time_of_year']:
        ds_th = dp.import_map_th(config)
    else:
        ds_th = None

    # Execute the main RHITA function
    main.main(ds, ds_th, config)

    # Log and print completion time
    end_time = t.time()
    execution_time = end_time - start_time
    logging.info(f"Application RHITA finished at {datetime.now()} in {execution_time:.1f} seconds")
    print(f"Application RHITA executed successfully in {execution_time:.1f} seconds")