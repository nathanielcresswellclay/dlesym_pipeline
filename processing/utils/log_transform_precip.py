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

    if not os.path.exists(config.output_filename) or config.overwrite:

        logger.info(f"Openning data from {config.input_filename}...")

        # open total precipitation data
        ds_tp6 = xr.open_dataset(config.input_filename, 
            chunks=OmegaConf.to_container(config.chunks))['total_precipitation']

        # if time slice is provided, select it
        if config.time is not None:
            ds_tp6 = ds_tp6.sel(time=slice(config.time.start, config.time.end))

        # enforce positive precipitation
        ds_tp6 = xr.where(ds_tp6 <= 0, 0, ds_tp6)
        # log transform precipitation
        ds_tp6 = np.log(ds_tp6 + config.epsilon) - np.log(config.epsilon)
        # rename variable
        ds_tp6 = ds_tp6.rename(f'tp6-lt{config.epsilon}')

        # save to netCDF in chunks
        logger.info(f'Saving log transformed precipitation to {config.output_filename}...')
        os.makedirs(os.path.dirname(config.output_filename), exist_ok=True)
        with ProgressBar():
            ds_tp6.to_netcdf(config.output_filename, compute=True, mode='w')
        logger.info('...Done!')
        ds_tp6.close()
    
    else:
        logger.info(f"File {config.output_filename} already exists and overwrite is set to False. Skipping computation.")

    if config.output_plot_dir:

        logger.info(f'Generating density plots for log transformed precipitation in {config.output_plot_dir} for time slice {config.plot_time_slice}...')
        # make dir if it does not exist
        os.makedirs(config.output_plot_dir, exist_ok=True)

        # load cached data for plotting
        ds_before = xr.open_dataarray(config.input_filename).sel(time=slice(config.plot_time_slice.start, config.plot_time_slice.end))
        ds_after = xr.open_dataarray(config.output_filename).sel(time=slice(config.plot_time_slice.start, config.plot_time_slice.end))

        # plot density plots
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        axs[0].hist(ds_before.values.flatten(), bins=100, label='P', alpha=0.5)
        axs[0].legend()
        axs[1].hist(ds_after.values.flatten(), bins=100, label=f'log(P + {config.epsilon}) - log({config.epsilon})', alpha=0.5)
        axs[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(config.output_plot_dir, 'density_plot.png'))
        plt.close(fig)
    return




if __name__=="__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Log transform precipitation and save to netCDF.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)

