"""
Compute tau300-700 from z300 and z700 components.

This script is driven by an OmegaConf/YAML configuration file (see
`processing/configs/calculate_tau300-700_dev.yaml` for an example).

It:
  1. Opens z300 and z700 NetCDF files as xarray datasets.
  2. Slices to `config.time.start` .. `config.time.end` (via `sel(time=slice(...))`).
  3. Computes tau300-700 as `z300 - z700`.
  4. Writes the result to `config.output_filename`.
  5. Optionally generates PNG frames for `config.plot_times` into `config.output_plot_dir`.
"""

import os
import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    """
    Parameters
    ----------
    config : omegaconf.DictConfig
        Expected fields:
          - `file_z300` (str): path to a NetCDF file containing variable `z300`
          - `file_z700` (str): path to a NetCDF file containing variable `z700`
          - `time.start` (str): ISO timestamp for the start of the slice
          - `time.end` (str): ISO timestamp for the end of the slice
          - `chunks.time` (int): dask chunk size along the `time` dimension
          - `overwrite` (bool): whether to recompute if `output_filename` exists
          - `plot_times` (list[str]): timestamps to render PNGs from the output
          - `output_plot_dir` (str): directory where PNGs are saved
          - `output_filename` (str): NetCDF file output path
    """

    if not os.path.exists(config.output_filename) or config.overwrite:

        logger.info(f"Opening data from {config.file_z300} and {config.file_z700}...")

        # resolve chunks 
        chunks = OmegaConf.to_container(config.chunks)
        # Open z300 and z700 component data
        ds_z300 = xr.open_dataset(config.file_z300,chunks=chunks)['geopotential']
        ds_z700 = xr.open_dataset(config.file_z700,chunks=chunks)['geopotential']
        # select time slice
        ds_z300 = ds_z300.sel(time=slice(config.time.start, config.time.end)).rename('tau300-700')
        ds_z700 = ds_z700.sel(time=slice(config.time.start, config.time.end)).rename('tau300-700')

        # calculate tau300-700 in chunks
        logger.info("Calculating tau300-700...")
        tau300_700 = ds_z300 - ds_z700
        tau300_700 = tau300_700.chunk(chunks)
        logger.info(f'Saving tau300-700 to {config.output_filename}...')
        os.makedirs(os.path.dirname(config.output_filename), exist_ok=True)
        with ProgressBar():
            tau300_700.to_netcdf(config.output_filename, mode='w')
        logger.info('...Done!')
    else:
        logger.info(f"File {config.output_filename} already exists and overwrite is set to False. Skipping computation.")

    # Plot frames for timestamps specified in config
    os.makedirs(config.output_plot_dir, exist_ok=True)
    tau300_700 = xr.open_dataset(config.output_filename)['tau300-700']
    for plot_time in tqdm(config.plot_times):
        plot_time = np.datetime64(plot_time)
        tau300_700_plot = tau300_700.sel(time=plot_time).load()
        fig, ax = plt.subplots(figsize=(8,6))
        im = tau300_700_plot.plot(ax=ax, cmap='viridis',
            cbar_kwargs={'shrink': 0.7, 'label': 'Tau300-700 (m/s)','orientation':'vertical'})
        plt.title(f'Tau300-700 at {plot_time}')
        plot_filename = os.path.join(config.output_plot_dir, f'tau300-700_{np.datetime_as_string(plot_time, unit="h")}.png')
        fig.savefig(plot_filename)
        plt.close()

    return

if __name__=="__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Calculate tau300-700 from z300 and z700 components and save to netCDF.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)

