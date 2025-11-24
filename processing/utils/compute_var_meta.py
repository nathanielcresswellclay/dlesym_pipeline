import os
import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _calculate_min(da: xr.DataArray, newname: str, output_file_prefix: str, overwrite: bool = False, output_plot_dir: str = None):
    """
    Calculate the minimum value of the DataArray and save to netCDF.
    params:
        da: xr.DataArray: input data array
        newname: str: new variable name (e.g., 'stl1_min')
        output_file_prefix: str: prefix for the output file name
        overwrite: bool: whether to overwrite existing files
        output_plot_dir: str: directory for output plots
    returns:
        None
    """
    
    # calculate min
    logger.info("Calculating mins temporally...")
    da_min = da.min(dim='time', skipna=True)

    # assign new new variable name 
    da_min = da_min.rename(newname)

    # save to netCDF
    output_filename = f"{output_file_prefix}_min.nc"
    if os.path.exists(output_filename) and not overwrite:
        logger.info(f"File {output_filename} already exists and overwrite is set to False. Skipping save.")
        return
    else:
        with ProgressBar():
            da_min.to_netcdf(f"{output_file_prefix}_min.nc", compute=True, mode='w')
        logger.info(f"Minimum values saved to {output_file_prefix}_min.nc")

    if output_plot_dir:
        # create output plot directory if it does not exist
        os.makedirs(output_plot_dir, exist_ok=True)

        # plot min values for a quick look
        fig, ax = plt.subplots(figsize=(10, 6))
        im = da_min.plot(ax=ax, cmap='viridis')
        plot_filename = os.path.join(output_plot_dir, f"min_plot.png")
        logger.info(f"Saving min plot to {plot_filename}")
        fig.savefig(plot_filename)
        plt.close()
        

def _calculate_max(da: xr.DataArray, newname: str, output_file_prefix: str, overwrite: bool = False, output_plot_dir: str = None):
    """
    Calculate the maximum value of the DataArray and save to netCDF.
    params:
        da: xr.DataArray: input data array
        newname: str: new variable name (e.g., 'stl1_max')
        output_file_prefix: str: prefix for the output file name
        overwrite: bool: whether to overwrite existing files
        output_plot_dir: str: directory for output plots
    returns:
        None
    """
    
    # calculate max
    logger.info("Calculating maxes temporally...")
    da_max = da.max(dim='time', skipna=True)

    # assign new new variable name 
    da_max = da_max.rename(newname)

    # save to netCDF
    output_filename = f"{output_file_prefix}_max.nc"
    if os.path.exists(output_filename) and not overwrite:
        logger.info(f"File {output_filename} already exists and overwrite is set to False. Skipping save.")
        return
    else:
        with ProgressBar():
            da_max.to_netcdf(f"{output_file_prefix}_max.nc", compute=True, mode='w')
        logger.info(f"Maximum values saved to {output_file_prefix}_max.nc")
    
    if output_plot_dir:
        # create output plot directory if it does not exist
        os.makedirs(output_plot_dir, exist_ok=True)

        # plot max values for a quick look
        fig, ax = plt.subplots(figsize=(10, 6))
        im = da_max.plot(ax=ax, cmap='viridis')
        plot_filename = os.path.join(output_plot_dir, f"max_plot.png")
        logger.info(f"Saving max plot to {plot_filename}")
        fig.savefig(plot_filename)
        plt.close()


    

def _calculate_annual_range(da: xr.DataArray, newname:str, output_file_prefix: str, overwrite: bool, output_plot_dir: str = None):
    """
    Calculate the annual range (max - min) of the DataArray and save to netCDF.
    params:
        da: xr.DataArray: input data array
        newname: str: new variable name (e.g., 'stl1_annual_range')
        output_file_prefix: str: prefix for the output file name
        overwrite: bool: whether to overwrite existing files
        output_plot_dir: str: directory for output plots
    returns:
        xr.DataArray: annual range data array
    """
    
    # calculate annual range
    logger.info("Calculating annual ranges temporally...")
    daily_climo = da.groupby('time.dayofyear').mean(dim='time', skipna=True)
    da_climo_min = daily_climo.min(dim='dayofyear', skipna=True)
    da_climo_max = daily_climo.max(dim='dayofyear', skipna=True)
    da_climo_range = da_climo_max - da_climo_min

    # assign new new variable name 
    da_climo_range = da_climo_range.rename(newname)

    # save to netCDF
    output_filename = f"{output_file_prefix}_annual_range.nc"
    if os.path.exists(output_filename) and not overwrite:
        logger.info(f"File {output_filename} already exists and overwrite is set to False. Skipping save.")
        return
    else:
        with ProgressBar():
            da_climo_range.to_netcdf(output_filename, compute=True, mode='w')
        logger.info(f"Annual range values saved to {output_filename}")
    
    if output_plot_dir:
        # create output plot directory if it does not exist
        os.makedirs(output_plot_dir, exist_ok=True)

        # plot annual range values for a quick look
        fig, ax = plt.subplots(figsize=(10, 6))
        im = da_climo_range.plot(ax=ax, cmap='viridis')
        plot_filename = os.path.join(output_plot_dir, f"annual_range_plot.png")
        logger.info(f"Saving annual range plot to {plot_filename}")
        fig.savefig(plot_filename)
        plt.close()

def main(config):

    # log config
    for file_cfg in config: 

        # get dict object from config
        file_cfg = OmegaConf.to_container(file_cfg)
        logger.info(f'Processing file: {file_cfg}')

        # create output directory recursively if it does not exist
        os.makedirs(os.path.dirname(file_cfg['output_file_prefix']), exist_ok=True)

        # load dataset
        da = xr.open_dataset(file_cfg['filename'], chunks=file_cfg.get('chunks', None))[file_cfg['variable']]
        # subset over time if specified
        logger.info(f'Calculating metas over time range: {file_cfg["time_slice"]}')
        da = da.sel(time=slice(file_cfg['time_slice']['start'], file_cfg['time_slice']['end']))

        # calculate min
        if 'min' in file_cfg['metas']:
            _calculate_min(
                da=da,
                newname=f"{file_cfg['variable']}_min",
                output_file_prefix=file_cfg['output_file_prefix'],
                overwrite=file_cfg.get('overwrite', False),
                output_plot_dir=file_cfg.get('output_plot_dir', None)
            )

        # calculate max
        if 'max' in file_cfg['metas']:
            _calculate_max(
                da=da,
                newname=f"{file_cfg['variable']}_max",
                output_file_prefix=file_cfg['output_file_prefix'],
                overwrite=file_cfg.get('overwrite', False),
                output_plot_dir=file_cfg.get('output_plot_dir', None)
            )

        # calculate annual range
        if 'annual_range' in file_cfg['metas']:
            _calculate_annual_range(
                da=da,
                newname=f"{file_cfg['variable']}_annual_range",
                output_file_prefix=file_cfg['output_file_prefix'],
                overwrite=file_cfg.get('overwrite', False),
                output_plot_dir=file_cfg.get('output_plot_dir', None)
            )
        
    logger.info("Variable metadata computation completed.")
    return

if __name__=="__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Compute variable metadata: min, max, annual range")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)
