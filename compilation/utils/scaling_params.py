import os
import shutil
import xarray as xr
import numpy as np
import omegaconf
import pandas as pd
from dask.diagnostics import ProgressBar
import logging
logging.basicConfig(level=logging.INFO)
from dask.diagnostics import ProgressBar
from dask.diagnostics import ProgressBar
logger = logging.getLogger(__name__)

def main(config: str):
    """
    calculate and log scaling parameters for channels in a dataset 
    Args:
        config (str): Path to the configuration file.
    Returns:
        None
    """

    # load config
    cfg = omegaconf.OmegaConf.load(config)
    logger.info(f"Loaded config {cfg}")

    # configure logging
    log_path = cfg.log_file
    # Reconfigure root logger
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO)

    ds = xr.open_zarr(cfg.input_file).sel(time=slice(cfg.time_range.start, cfg.time_range.end))

    # temporary file to store scaling parameters 
    temp_file=cfg.log_file.replace('.log', '_var.nc')
    logger.info(f"Loaded dataset {ds}")

    # log
    logger.info(f"Scaling parameters calculated and saved to {temp_file}")

    # inputs 
    inputs_mean = ds.inputs.mean(dim=('time', 'width', 'height', 'face'), skipna=True)
    inputs_std = ds.inputs.std(dim=('time', 'width', 'height', 'face'), skipna=True)
    with ProgressBar():
        inputs_mean.to_netcdf(temp_file.replace('var.nc', f'inputs_mean.nc'))
        inputs_std.to_netcdf(temp_file.replace('var.nc', f'inputs_std.nc'))
    inputs_mean.close()
    inputs_std.close()

    # targets
    targets_mean = ds.targets.mean(dim=('time', 'width', 'height', 'face'), skipna=True)
    targets_std = ds.targets.std(dim=('time', 'width', 'height', 'face'), skipna=True)
    with ProgressBar():
        targets_mean.to_netcdf(temp_file.replace('var.nc', f'targets_mean.nc'))
        targets_std.to_netcdf(temp_file.replace('var.nc', f'targets_std.nc'))
    targets_mean.close()
    targets_std.close()

    # constants
    constants_mean = ds.constants.mean(dim=('width', 'height', 'face'), skipna=True)
    constants_std = ds.constants.std(dim=('width', 'height', 'face'), skipna=True)
    with ProgressBar():
        constants_mean.to_netcdf(temp_file.replace('var.nc', f'constants_mean.nc'))
        constants_std.to_netcdf(temp_file.replace('var.nc', f'constants_std.nc'))
    constants_mean.close()
    constants_std.close()
    
    # log contents of saved datasets 
    inputs_mean = xr.open_dataset(temp_file.replace('var.nc', f'inputs_mean.nc')).inputs
    inputs_std = xr.open_dataset(temp_file.replace('var.nc', f'inputs_std.nc')).inputs
    targets_mean = xr.open_dataset(temp_file.replace('var.nc', f'targets_mean.nc')).targets
    targets_std = xr.open_dataset(temp_file.replace('var.nc', f'targets_std.nc')).targets
    constants_mean = xr.open_dataset(temp_file.replace('var.nc', f'constants_mean.nc')).constants
    constants_std = xr.open_dataset(temp_file.replace('var.nc', f'constants_std.nc')).constants
    
    for channel in inputs_mean.channel_in.values:
        logger.info(f"Scaling parameter for input channel {channel}:")
        logger.info(f"    mean: {inputs_mean.sel(channel_in=channel).values}")
        logger.info(f"    std: {inputs_std.sel(channel_in=channel).values}")
    for channel in targets_mean.channel_out.values:
        logger.info(f"Scaling parameter for target channel {channel}:")
        logger.info(f"    mean: {targets_mean.sel(channel_out=channel).values}")
        logger.info(f"    std: {targets_std.sel(channel_out=channel).values}")
    for channel in constants_mean.channel_c.values:
        logger.info(f"Scaling parameter for constant channel {channel}:")
        logger.info(f"    mean: {constants_mean.sel(channel_c=channel).values}")
        logger.info(f"    std: {constants_std.sel(channel_c=channel).values}")
    


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Calculate mean and std for channels in the training set.")
    parser.add_argument('--config', type=str, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)
