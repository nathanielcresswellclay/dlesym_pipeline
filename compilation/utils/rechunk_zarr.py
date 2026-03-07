import os
import shutil
import xarray as xr
import numpy as np
import omegaconf
import pandas as pd
from dask.diagnostics import ProgressBar
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config: str):
    """
    rechunk zarr file based on the provided configuration.
    Args:
        config (str): Path to the configuration file.
    Returns:
        None
    """

    # load config
    cfg = omegaconf.OmegaConf.load(config)
    logger.info(f"Loaded config {cfg}")

    # check if output file already exists
    if os.path.exists(cfg.output_file):
        if not cfg.overwrite:
            logger.info(f'Output file {cfg.output_file} already exists. Set overwrite=True to overwrite it.')
            return
        else:
            logger.info(f'Output file {cfg.output_file} already exists. Overwriting it.')
            shutil.rmtree(cfg.output_file)
    
    # open dataset and chunk
    ds = xr.open_zarr(cfg.input_file)
    logger.info(f'dataset: {ds.inputs}')
    ds =ds.chunk(cfg.chunks)
    logger.info(f"Rechunked dataset: {ds.inputs}")

    # update channel_in, channel_out, channel_c dims to be strings
    ds = ds.assign_coords(channel_in=ds['channel_in'].astype(str),
                    channel_out=ds['channel_out'].astype(str),
                    channel_c=ds['channel_c'].astype(str))

    # update encoding so to_zarr uses rechunked sizes (encoding from source zarr
    # otherwise overrides Dask chunks and writes with original chunk sizes)
    for var in ds.data_vars:
        if hasattr(ds[var].data, 'chunks'):
            ds[var].encoding['chunks'] = tuple(
                c[0] for c in ds[var].data.chunks
            )
    logger.info(f'Encoding: {ds.encoding}')
    # check header 
    logger.info(f'Rechunked dataset: {ds}') 

    # rechunk
    logger.info(f'Saving rechunked dataset to {cfg.output_file}...')
    if cfg.in_memory:

        # load data into memory
        logger.info("Loading dataset into memory before rechunking...")
        ds = ds.compute()
        logger.info("Dataset loaded into memory.")

        # save to zarr
        logger.info(f"Saving rechunked dataset to {cfg.output_file}...")
        ds.to_zarr(cfg.output_file, mode='w')
        logger.info(f"Dataset saved successfully.")

    else:

        # save in chunks
        logger.info(f"Saving dataset in chunks to {cfg.output_file}...")
        # save to zarr
        with ProgressBar():
            ds.to_zarr(cfg.output_file, mode='w', consolidated=True)
        logger.info(f"Dataset saved successfully.")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Rechunk a Zarr file based on the provided configuration.")
    parser.add_argument('--config', type=str, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)
