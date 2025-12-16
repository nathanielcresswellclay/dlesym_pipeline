import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from tqdm import tqdm
from omegaconf import OmegaConf
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def impute_maiac(
    filename: str,
    variable: str,
    constant_value: float,
    output_file: str,
    chunks: dict = {'time': 8},
    overwrite: bool = False,
    time_slice = None,
    plotting: dict = None,
):
    """
    Linear interpolation in time to fill missing NDVI data followed by constant value imputation over ocean.
    
    Args:
        filename (str): Path to the input netCDF file.
        variable (str): Name of the variable to impute.
        constant_value (float): Constant value to use for imputation over ocean.
        output_file (str): Path to save the output netCDF file. If None, overwrites the input file.
        chunks (dict): Chunk sizes for the data loading.
        overwrite (bool): Whether to overwrite the output file if it exists.
        time_slice (slice): Optional time slice to select a subset of the data.
        plotting (dict): Optional dictionary for plotting configurations.
    Returns:
        None
    """

    # check if output file exists
    if os.path.exists(output_file) and not overwrite:
        logger.info(f'Output file {output_file} exists and overwrite is False, skipping imputation...')
        return
    # make output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # load dataset, selecting time if specified
    ds = xr.open_dataset(filename, chunks='auto')[variable]
    # subsample if time_slice is specified
    if time_slice is not None:
        logger.info(f'Selecting time slice: {time_slice}')
        ds = ds.sel(time=slice(time_slice['start'], time_slice['end']))
    else:  
        logger.info(f'No time slice specified, using full time range: {ds.time.values[0]} to {ds.time.values[-1]}') 

    # load dataset into memory
    logger.info("Loading dataset into memory...")
    ds.load()
    logger.info("Dataset loaded into memory.")

    # impute in time. 
    logger.info("Performing linear interpolation in time to fill missing data...")
    ds = ds.interpolate_na(dim='time', method='linear')
    logger.info("Time interpolation complete.")

    # impute
    ds = ds.fillna(constant_value)

    # finally we rechunk to original chunks and save to final output file
    logger.info(f'Rechunking to original chunks and saving final output to {output_file}...')
    ds = ds.chunk(chunks)
    logger.info('Saving final output...')
    ds.to_netcdf(output_file, mode='w')
    logger.info(f'Final output saved to {output_file}')
    logger.info(ds)
    # release memory
    ds.close()


    # plot single frame. for quick visualization during dev and debugging
    if plotting is not None:
        logger.info('Generating quick plots for imputed data...')
        # make output dir if it doesn't exist
        os.makedirs(plotting['output_dir'], exist_ok=True)
        # load original data for comparison
        ds_before = xr.open_dataset(filename, chunks='auto')[variable]
        ds_after = xr.open_dataset(output_file, chunks='auto')[variable]
        pbar = tqdm(plotting['times'], desc='Generating quick plots')
        for time in pbar:
            pbar.set_description(f'Plotting {str(time)[:10]}')
            try:
                temp_data_before = ds_before.sel(time=np.datetime64(time))
                temp_data_before.load()
                temp_data_after = ds_after.sel(time=np.datetime64(time))
                temp_data_after.load()
            except Exception as e:
                logger.warning(f"Failed to load data for time {time}: {e}")
                continue
            fig, axes = plt.subplots(1, 2, figsize=(15,7), subplot_kw={'projection': ccrs.PlateCarree()})
            temp_data_before.plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'label': variable})
            temp_data_after.plot(ax=axes[1], transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'label': variable})
            axes[0].coastlines()
            axes[1].coastlines()
            axes[0].set_title(f'Before Impute {variable} on {str(time)[:13]}')
            axes[1].set_title(f'After Impute {variable} on {str(time)[:13]}')
            plt.savefig(os.path.join(plotting['output_dir'], f'{str(time)[:13]}.png'))
            plt.close(fig)
            temp_data_after.close()
            temp_data_before.close()
        
def impute_maiac_parallel(
    filename: str,
    variable: str,
    constant_value: float,
    output_file: str,
    chunks: dict = {'time': 8},
    overwrite: bool = False,
    time_slice = None,
    plotting: dict = None,
):
    """
    Linear interpolation in time to fill missing NDVI data followed by constant value imputation over ocean.
    
    Args:
        filename (str): Path to the input netCDF file.
        variable (str): Name of the variable to impute.
        constant_value (float): Constant value to use for imputation over ocean.
        output_file (str): Path to save the output netCDF file. If None, overwrites the input file.
        chunks (dict): Chunk sizes for the data loading.
        overwrite (bool): Whether to overwrite the output file if it exists.
        time_slice (slice): Optional time slice to select a subset of the data.
        plotting (dict): Optional dictionary for plotting configurations.
    Returns:
        None
    """

    # check if output file exists
    if os.path.exists(output_file) and not overwrite:
        logger.info(f'Output file {output_file} exists and overwrite is False, skipping imputation...')
        return
    # make output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # load dataset, selecting time if specified
    ds = xr.open_dataset(filename, chunks='auto')[variable]
    # subsample if time_slice is specified
    if time_slice is not None:
        logger.info(f'Selecting time slice: {time_slice}')
        ds = ds.sel(time=slice(time_slice['start'], time_slice['end']))
    else:  
        logger.info(f'No time slice specified, using full time range: {ds.time.values[0]} to {ds.time.values[-1]}') 


    # rechunk along lat for efficient time interpolation
    logger.info("Rechunking data along latitude for time interpolation...")
    ds = ds.chunk({'latitude': 1, 'time':-1})

    # save chunked data to temporary file
    output_file_lat_chunk = output_file.replace('.nc', '_lat_chunk.nc')
    logger.info(f'Saving lat-chunked data to temporary file {output_file_lat_chunk}...')
    logger.warning('using Dask chunks and temporary files for impute. May fail in the case of large numbers of small chunks due to dask overhead.')
    with ProgressBar():
        ds.to_netcdf(output_file_lat_chunk, compute=True, mode='w')
    logger.info(f'Lat rechunking complete.')
    # release memory
    ds.close()

    # reload chunked data
    ds = xr.open_dataset(output_file_lat_chunk)[variable]

    # impute in time. 
    ds_time_impute = ds.interpolate_na(dim='time', method='linear')

    # save in chunks 
    output_file_time_impute = output_file.replace('.nc', '_time_imputed.nc')
    logger.info(f'Saving time-imputed data to temporary file {output_file_time_impute}...')
    with ProgressBar():
        ds_time_impute.to_netcdf(output_file_time_impute, compute=True, mode='w')
    logger.info(f'Time imputation complete.')
    # release memory
    ds.close()
    ds_time_impute.close()

    # reload time imputed data with original chunks
    ds = xr.open_dataset(output_file_time_impute)[variable]

    # impute
    ds_imputed = ds.fillna(constant_value)
    output_file_constant_impute = output_file.replace('.nc', '_constant_imputed.nc')
    logger.info(f'Running constant imputation and saving output in {output_file_constant_impute}...')
    with ProgressBar():
        ds_imputed.to_netcdf(output_file_constant_impute, compute=True)
    logger.info(f'Imputation complete. Output saved to {output_file_constant_impute}')
    # release memory
    ds.close()
    ds_imputed.close()

    # finally we rechunk to original chunks and save to final output file
    ds = xr.open_dataset(output_file_constant_impute)[variable]
    ds_imputed = ds.chunk(chunks)
    logger.info(f'Rechunking to original chunks and saving final output to {output_file}...')
    logger.warning('using Dask chunks and temporary files for final output. May fail in the case of large numbers of small chunks due to dask overhead.')
    with ProgressBar():
        ds_imputed.to_netcdf(output_file, compute=True, mode='w')
    logger.info(f'Final output saved to {output_file}')
    logger.info(ds_imputed)
    # release memory
    ds.close()
    ds_imputed.close()

    # plot single frame. for quick visualization during dev and debugging
    if plotting is not None:
        logger.info('Generating quick plots for imputed data...')
        # make output dir if it doesn't exist
        os.makedirs(plotting['output_dir'], exist_ok=True)
        pbar = tqdm(plotting['times'], desc='Generating quick plots')
        for time in pbar:
            pbar.set_description(f'Plotting {str(time)[:10]}')
            try:
                temp_data_before = ds.sel(time=np.datetime64(time))
                temp_data_before.load()
                temp_data_after = ds_imputed.sel(time=np.datetime64(time))
                temp_data_after.load()
            except Exception as e:
                logger.warning(f"Failed to load data for time {time}: {e}")
                continue
            fig, axes = plt.subplots(1, 2, figsize=(15,7), subplot_kw={'projection': ccrs.PlateCarree()})
            temp_data_before.plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'label': variable})
            temp_data_after.plot(ax=axes[1], transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'label': variable})
            axes[0].coastlines()
            axes[1].coastlines()
            axes[0].set_title(f'Before Impute {variable} on {str(time)[:13]}')
            axes[1].set_title(f'After Impute {variable} on {str(time)[:13]}')
            plt.savefig(os.path.join(plotting['output_dir'], f'{str(time)[:13]}.png'))
            plt.close(fig)
            temp_data_after.close()
            temp_data_before.close()
        
    logger.info('Cleaning up temporary files...')
    os.remove(output_file_lat_chunk)
    os.remove(output_file_time_impute)
    os.remove(output_file_constant_impute)

    # clean up temporary files

def main(config: OmegaConf):

    for file_cfg in config: 

        logger.info(f'Imputing constant value over ocean. Config: {file_cfg}')
        impute_maiac(
            filename=file_cfg['filename'],
            variable=file_cfg['variable'],
            constant_value=file_cfg['constant_value'],
            output_file=file_cfg['output_file'],
            chunks=file_cfg.get('chunks', 'auto'),
            overwrite=file_cfg.get('overwrite', False),
            time_slice=file_cfg.get('time_slice', None),
            plotting=file_cfg.get('plotting', None),
        )

if __name__=="__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Impute MAIAC data with temporal interpolation and constant values over ocean ")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)
