"""
Compute 10m windspeed magnitude (ws10) from u10/v10 components.

This script is driven by an OmegaConf/YAML configuration file (see
`processing/configs/calculate_windspeed_dev.yaml` for an example).

It:
  1. Opens u10 and v10 NetCDF files as xarray datasets.
  2. Slices to `config.time.start` .. `config.time.end` (via `sel(time=slice(...))`).
  3. Computes windspeed as `sqrt(u10**2 + v10**2)`.
  4. Writes the result to `config.output_filename`.
  5. Optionally generates PNG frames for `config.plot_times` into `config.output_plot_dir`.
"""

import os
import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from dask.diagnostics import ProgressBar
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
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
          - `u_filename` (str): path to a NetCDF file containing variable `u10`
          - `v_filename` (str): path to a NetCDF file containing variable `v10`
          - `time.start` (str): ISO timestamp for the start of the slice
          - `time.end` (str): ISO timestamp for the end of the slice
          - `chunks.time` (int): dask chunk size along the `time` dimension
          - `overwrite` (bool): whether to recompute if `output_filename` exists
          - `plot_times` (list[str]): timestamps to render PNGs from the output
          - `output_plot_dir` (str): directory where PNGs are saved
          - `output_filename` (str): NetCDF file output path
    """

    if not os.path.exists(config.output_filename) or config.overwrite:

        logger.info(f"Opening data from {config.u_filename} and {config.v_filename}...")

        # resolve chunks 
        chunks = OmegaConf.to_container(config.chunks)
        # Open u10 and v10 component data
        ds_u = xr.open_dataset(config.u_filename,chunks=chunks)['10m_u_component_of_wind']
        ds_v = xr.open_dataset(config.v_filename,chunks=chunks)['10m_v_component_of_wind']
        # select time slice
        ds_u = ds_u.sel(time=slice(config.time.start, config.time.end)).rename('ws10')
        ds_v = ds_v.sel(time=slice(config.time.start, config.time.end)).rename('ws10')

        # calculate windspeed in chunks
        logger.info("Calculating windspeed...")
        # `ws10` is a scalar magnitude field computed from the u/v components.
        windspeed = np.sqrt(ds_u**2 + ds_v**2)
        windspeed = windspeed.chunk(chunks)
        logger.info(f'Saving windspeed to {config.output_filename}...')
        os.makedirs(os.path.dirname(config.output_filename), exist_ok=True)
        with ProgressBar():
            windspeed.to_netcdf(config.output_filename, mode='w')
        logger.info('...Done!')
        
    else:
        logger.info(f"File {config.output_filename} already exists and overwrite is set to False. Skipping computation.")

    # Plot frames for timestamps specified in config
    os.makedirs(config.output_plot_dir, exist_ok=True)
    windspeed = xr.open_dataset(config.output_filename)['ws10']
    for plot_time in tqdm(config.plot_times):
        plot_time = np.datetime64(plot_time)
        windspeed_plot = windspeed.sel(time=plot_time).load()
        fig, ax = plt.subplots(figsize=(8,6))
        im = windspeed_plot.plot(ax=ax, cmap='viridis',
            cbar_kwargs={'shrink': 0.7, 'label': 'Windspeed (m/s)','orientation':'vertical'})
        plt.title(f'Windspeed at {plot_time}')
        plot_filename = os.path.join(config.output_plot_dir, f'windspeed_{np.datetime_as_string(plot_time, unit="h")}.png')
        fig.savefig(plot_filename)
        plt.close()

    return

if __name__=="__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Calculate windspeed from u and v components and save to netCDF.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)

