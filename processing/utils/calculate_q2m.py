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

def add_q2m(sp, td2m):
    """Calculate 2-metre specific humidity from surface pressure and t2m."""
    q2m = specific_humidity_from_dewpoint(sp * units.Pa, td2m * units.degK)
    return q2m

def main(config):

    if not os.path.exists(config.output_filename) or config.overwrite:

        logger.info(f"Openning data from {config.td2m_filename} and {config.sp_filename}...")

        # open t2m and sp data
        ds_td2m = xr.open_dataset(config.td2m_filename)['2m_dewpoint_temperature']
        ds_sp = xr.open_dataset(config.sp_filename)['surface_pressure']
        
        # select time slice
        ds_td2m = ds_td2m.sel(time=slice(config.time.start, config.time.end))
        ds_sp = ds_sp.sel(time=slice(config.time.start, config.time.end))

        # chunk data for dask processing
        ds_td2m = ds_td2m.chunk(config.chunks)
        ds_sp = ds_sp.chunk(config.chunks)

        # calculate q2m
        logger.info("Calculating 2-metre specific humidity...")
        q2m = add_q2m(ds_sp, ds_td2m)
        # add variable name 
        q2m = q2m.rename('q2m')

        # enforce chunks on output
        q2m = q2m.chunk(config.chunks)

        # save to netCDF in chunks
        logger.info(f'Saving 2-metre specific humidity to {config.output_filename}...')
        os.makedirs(os.path.dirname(config.output_filename), exist_ok=True)
        with ProgressBar():
            q2m.to_netcdf(config.output_filename, compute=True, mode='w')
        logger.info('...Done!')
    
    else:
        logger.info(f"File {config.output_filename} already exists and overwrite is set to False. Skipping computation.")

    if config.output_plot_dir:

        logger.info(f'Generating {len(config.plot_times)} plots for q2m in {config.output_plot_dir}...')
        # make dir if it does not exist
        os.makedirs(config.output_plot_dir, exist_ok=True)

        # load cached data for plotting
        q2m = xr.open_dataarray(config.output_filename)

        for plot_time in tqdm(config.plot_times):
            plot_time = np.datetime64(plot_time)
            q2m_plot = q2m.sel(time=plot_time).load()
            fig, ax = plt.subplots(figsize=(8,6))
            im = q2m_plot.plot(ax=ax, cmap='viridis',
                cbar_kwargs={'shrink': 0.7, 'label': '2m Specific Humidity (kg/kg)','orientation':'vertical'})
            plt.title(f'2-metre Specific Humidity at {plot_time}')
            plot_filename = os.path.join(config.output_plot_dir, f'q2m_{np.datetime_as_string(plot_time, unit="h")}.png')
            fig.savefig(plot_filename)
            plt.close()
    return




if __name__=="__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Calculate 2-metre specific humidity and save to netCDF.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)

