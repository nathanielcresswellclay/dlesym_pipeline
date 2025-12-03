import os
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime, timedelta
from dask.diagnostics import ProgressBar
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def maiac_filename_to_datetime64(fname: str) -> np.datetime64:
    """
    Parse MAIAC / MODIS filenames of the form:
        MCD19A3CMG.AYYYYDDD.*.hdf
    and return the observation day as np.datetime64[ns].
    """
    # Split and find the part starting with 'A'
    parts = fname.split('.')
    date_token = None
    for p in parts:
        if p.startswith('A') and len(p) >= 8:
            date_token = p  # AYYYYDDD
            break
    if date_token is None:
        raise ValueError(f"Could not find AYYYYDDD portion in filename: {fname}")

    year = int(date_token[1:5])
    doy  = int(date_token[5:8])

    # Convert year + day-of-year → datetime
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)

    return np.datetime64(dt, 'ns')

def main(config):

    # check if output_file exists already
    if os.path.exists(config.output_file) and not config.overwrite:
        logger.info(f'Output file {config.output_file} already exists. Exiting.')
        return

    ###########################################################################
    # Load raw data files and resolve time

    # get list of files inside raw data directory
    raw_files = [os.path.join(config.raw_data_dir, f) for f in os.listdir(config.raw_data_dir) if f.endswith('.hdf')]
    # extract datetime from filenames
    file_datetimes = np.array([maiac_filename_to_datetime64(os.path.basename(f)) for f in raw_files])

    # define target time coordinate from parameters
    target_datetimes = pd.date_range(start=config.time_range_params.start_date,
                                 end=config.time_range_params.end_date,
                                 freq=config.time_range_params.freq)
    
    # find files that fall within the target time range
    target_file_intersect = np.isin(file_datetimes, target_datetimes.values)
    logger.info(f'Found {len(raw_files)} raw files in {config.raw_data_dir}...')
    logger.info(f'{np.sum(target_file_intersect)} files fall within target time range {config.time_range_params.start_date} to {config.time_range_params.end_date}.')   

    # select only those files
    raw_files = np.array(raw_files)[target_file_intersect].tolist()
    file_datetimes = file_datetimes[target_file_intersect]
    # open as multi-file xarray dataset
    ds = xr.open_mfdataset(raw_files, concat_dim='time', combine='nested', engine='netcdf4', chunks={'time': config.chunks})[config.target_var]

    # assign time coordinates from filenames, enforce chronology
    ds = ds.assign_coords(time=('time', file_datetimes))
    ds = ds.sortby('time') # unfortunately, necessary

    ###########################################################################
    # Formatting grid

    # assign lat-lon coordinates. Assumes files on .05 degree lat-lon grid
    lat_vals = np.arange(90.0, -90.0, -0.05)
    lon_vals = np.arange(-180.0, 180.0, 0.05)
    # shift lon to 0-360
    lon_vals = (lon_vals + 360) % 360
    ds = ds.rename({'YDim:CMGgrid': 'latitude', 'XDim:CMGgrid': 'longitude'})
    ds = ds.assign_coords({'latitude': ('latitude', lat_vals), 'longitude': ('longitude', lon_vals)})

    # open_reference dataset for regridding
    logger.info(f'Opening reference dataset for regridding: {config.reference_file}')
    ref_ds = xr.open_dataset(config.reference_file)
    logger.info(f'Regridding to reference dataset grid with shape {ref_ds.dims}')
    ds = ds.interp(latitude=ref_ds.latitude, longitude=ref_ds.longitude, method='nearest') # interp call sorts coords automatically

    # enforce chunking for parallel write 
    ds = ds.chunk(time=config.chunks)
    # Save processed dataset to netCDF
    logger.info(f'Saving regridded file to temporary dataset: {config.output_file.replace(".nc", "temp.nc")}')

    with ProgressBar():
        ds.to_netcdf(config.output_file.replace('.nc', 'temp.nc'), mode='w')
    # cleanup
    ds.close()
    ref_ds.close()

    #########################################################################
    # super sample regridded data to target time grid

    ds = xr.open_dataset(config.output_file.replace('.nc', 'temp.nc'), chunks={'time': config.chunks})[config.target_var]
    logger.info(f'Supersampling regridded data with {len(ds.time)} timestamps to target time grid with {len(target_datetimes)} timestamps.')

    # interpolate dataset to target time dimension
    ds_resampled = ds.interp(time=target_datetimes.values, method='nearest')

    # enforce chunking and save in final output file
    ds_resampled = ds_resampled.chunk(time=config.chunks)
    logger.info(f'Saving regridded+supersampled dataset to final output file: {config.output_file}')
    with ProgressBar():
        ds_resampled.to_netcdf(config.output_file, mode='w')
    logger.info('Supersampling completed successfully.')
    logger.info(f'Saved dataset: {ds_resampled}')
    ds_resampled.close()

    # remove temporary file
    os.remove(config.output_file.replace('.nc', 'temp.nc'))

    ###########################################################################
    # Optional: plot a quick figure of the data

    if getattr(config, 'plotting', False):
        # make plotting directory if it doesn't exist
        os.makedirs(config.plotting.output_dir, exist_ok=True)
        # open final dataset
        ds = xr.open_dataset(config.output_file, chunks={'time': config.chunks})[config.target_var]
        # plot times 
        pbar = tqdm(config.plotting.times, desc='Generating quick plots')
        for time in pbar:
            pbar.set_description(f'Plotting {str(time)[:10]}')
            temp_data = ds.sel(time=np.datetime64(time))
            temp_data.load()
            fig = plt.figure(figsize=(10,5))
            ax = plt.axes(projection=ccrs.PlateCarree())
            temp_data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='BrBG', cbar_kwargs={'label': config.target_var})
            ax.coastlines()
            ax.set_title(f'{config.target_var} on {str(ds.time.values[0])[:10]}')
            plt.savefig(os.path.join(config.plotting.output_dir, f'{str(time)[:13]}.png'))
            plt.close(fig)
            temp_data.close()



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