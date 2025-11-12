import os 
import argparse
import omegaconf
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _era5_check(retrieval: omegaconf.DictConfig):

    ds = xr.open_dataarray(retrieval.file_path)
    logger.info(f'Checking file: {retrieval.file_path}')
    pbar= tqdm(ds.time.values)
    nan_count = 0
    global_average = []
    for t in pbar:
        pbar.set_description(f'Checking time {pd.to_datetime(t)}')
        ds_time = ds.sel(time=t).load()
        nan_count += int(ds_time.isnull().sum().values)
        global_average.append(float(ds_time.mean().values))
    logger.info(f'Total NaNs: {nan_count}')

    # plot global average time series
    logger.info(f'Plotting global average time series...')
    plt.figure(figsize=(12,6))
    plt.plot(ds.time.values, global_average, label='Global Average')
    plt.xlabel('Time')
    plt.savefig(retrieval.file_path.replace('.nc','_global_average_timeseries.png'))
    plt.close()

    # plot spatial map at specified time
    if 'plot_times' in retrieval.keys():
        pbar= tqdm(retrieval.plot_times)
        for plot_time in pbar:
            pbar.set_description(f'Plotting spatial map at time {plot_time}')
            plot_time = pd.to_datetime(plot_time)
            logger.info(f'Plotting spatial map at time {plot_time}...')
            ds_plot = ds.sel(time=plot_time).load()
            plt.figure(figsize=(8,6))
            ds_plot.plot(cmap='viridis')
            plt.title(f'Spatial Map at {plot_time}')
            plt.savefig(retrieval.file_path.replace('.nc',f'_spatial_map_{plot_time.strftime("%Y%m%dT%H%M")}.png'))
            plt.close()
    return 

def main(config: omegaconf.DictConfig):

    for retrival in config.retreivals:
        logger.info(f'Checking retrieval. File: {retrival.file_path}')
        source = retrival.source
        if source == 'era5':
            _era5_check(retrival)
        else:
            logger.error(f'Unknown source {source} specified in config.')

    return 
if __name__ == "__main__":

    # if run as script, accept config and pass to main as omegaconf DictConfig
    parser = argparse.ArgumentParser(description="Check retrieval for reasonableness.")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file.')
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(args.config)
    main(cfg)