import os
import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from datetime import datetime, timedelta
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
from processing.utils.processing_maiac import maiac_filename_to_datetime64
logger = logging.getLogger(__name__)

def main(config):

    # open download config to get parameters of retieval
    download_config = OmegaConf.load(config.download_config)

    # get list of files inside raw data directory
    downloads = [os.path.join(download_config.output_dir, f) for f in os.listdir(download_config.output_dir) if f.endswith('.hdf')]
    logger.info(f'Found {len(downloads)} downloaded files in {download_config.output_dir}.')

    # infer expected files from download config
    expected_times = pd.date_range(start=download_config.start_date, end=download_config.end_date, freq='D')
    logger.info(f'Expected number of files based on date range and daily samples: {len(expected_times)}.')

    # discrepency
    if len(downloads) != len(expected_times):
        num_missing = len(expected_times) - len(downloads)
        downloaded_times = np.array([maiac_filename_to_datetime64(os.path.basename(f)) for f in downloads])
        logger.warning(f'Discrepency between downloaded files and expected files: Missing {num_missing} files.')
        logger.warning('Missing dates:')
        for expected_time in expected_times:
            if np.datetime64(expected_time) not in downloaded_times:
                logger.warning(f' - {expected_time.date().isoformat()}')
    else:
        logger.info('Number of downloaded files matches expected number of files.')

if __name__=="__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Process MAIAC raw data files.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)