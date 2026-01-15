import os
import xarray as xr
import time
import numpy as np
import omegaconf
import pandas as pd
from dask.diagnostics import ProgressBar
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config: str):
    """
    Compile separate trainsets in a single Zarr file based on the provided configuration.    
    Args:
        config (str): Path to the configuration file.
    Returns:
        None
    """
    # load config as OmegaConf object
    cfg = omegaconf.OmegaConf.load(config)

    # check if output file already exists
    if os.path.exists(cfg.output_file) and not cfg.overwrite:
        logger.info(f'Output file {cfg.output_file} already exists. Set overwrite=True to overwrite it.')
        return

    # resolve newtime subset
    time_subset = pd.date_range(
        start=cfg.time_subset_start,
        end=cfg.time_subset_end,
        freq=cfg.time_subset_freq
    ).to_numpy().astype(np.datetime64)

    logger.info(f"Selecting common time subset: {time_subset}")
    # inputs
    inputs = []
    for input_file in cfg.input_files:

        # open file and extract desired vars
        ds = xr.open_zarr(input_file.path, chunks=None).inputs.sel(channel_in=input_file.channel_in)
        # select time dim
        ds = ds.sel(time=time_subset)
        inputs.append(ds)

    # targets
    targets = []
    for output_file in cfg.output_files:

        # open file and extract desired vars
        ds = xr.open_zarr(output_file.path, chunks=None).targets.sel(channel_out=output_file.channel_out)
        # select time dim
        ds = ds.sel(time=time_subset)
        targets.append(ds)

    #constants
    constants = []
    for constant_file in cfg.constant_files:

        # open file and extract desired vars
        ds = xr.open_zarr(constant_file.path, chunks=None).constants.sel(channel_c=constant_file.channel_c)
        constants.append(ds)

    # get list of coordinates 
    # concatenate dataarrays along channel dimension, enforce chunking
    logger.info(f"Concatenating datasets along channel dimensions..")
    logger.warning(f'Overriding coordinates to minimal for concatenation, resulting dataset will inherit coordinates from the first input dataset.')
    inputs = xr.concat(inputs, dim='channel_in', coords='minimal', compat='override')
    targets = xr.concat(targets, dim='channel_out', coords='minimal', compat='override')
    constants = xr.concat(constants, dim='channel_c', coords='minimal', compat='override')

    # enforce chunking
    if cfg.chunks is not None:
        logger.info(f"Chunking dataset: {cfg.chunks}")
        t0 = time.perf_counter()
        inputs = inputs.chunk(cfg.chunks)
        targets = targets.chunk(cfg.chunks)
        constants = constants.chunk({k: -1 for k in constants.dims})
        logger.info(f"Finished chunking in {time.perf_counter() - t0:.2f} seconds.")

    # create a new dataset with inputs, targets, and constants
    logger.info(f"Merging data arrays into a single dataset...")
    t0 = time.perf_counter()
    ds_merged = xr.Dataset({
        'inputs': inputs,
        'targets': targets,
        'constants': constants
    })
    logger.info(f"Merged dataset in {time.perf_counter() - t0:.2f} seconds: {ds_merged}")

    # enforce string encoding for channel names
    ds_merged['channel_in'] = ds_merged['channel_in'].astype(str)
    ds_merged['channel_out'] = ds_merged['channel_out'].astype(str) 
    ds_merged['channel_c'] = ds_merged['channel_c'].astype(str)

    # Clear inherited Zarr encoding so chunks come from Dask
    for v in ds_merged.data_vars:
        ds_merged[v].encoding.clear()

    # save to zarr
    logger.info(f'Saving merged dataset to {cfg.output_file}...')
    t0 = time.perf_counter()
    with ProgressBar():
        ds_merged.to_zarr(cfg.output_file, mode='w')
    logger.info(f"Dataset saved successfully to {cfg.output_file} in {time.perf_counter() - t0:.2f} seconds.")
    return
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compile seperate trainsets in a single zarr file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load the configuration
    main(config=args.config)