# super sample data e.g. daily to 3-hourly
import numpy as np
import xarray as xr
import omegaconf
import logging
import pandas as pd
from dask.diagnostics import ProgressBar
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def supersample(
        input_dataset: str,
        output_dataset: str,
        new_time_dim: np.ndarray,
):
    return

def main(config):
    """
    Main function to supersample data to a new time dimension.
    
    Parameters:
    config (dict): Configuration dictionary containing input and output dataset paths and new time dimension.
    """
    # Load the input dataset
    cfg = omegaconf.OmegaConf.load(config)

    times = pd.date_range(
        start=cfg.time_start,
        end=cfg.time_end,
        freq=cfg.time_freq
    ).to_numpy().astype('datetime64[ns]')

    # open the input dataset
    logger.info(f'Opening input dataset: {cfg.input_dataset}')
    ds = xr.open_dataset(cfg.input_dataset)

    # supersample the dataset
    logger.info('Resampling data using nearest available values...')
    ds = ds.resample(time=cfg.time_freq).nearest().sel(time=slice(cfg.time_start, cfg.time_end))
    # enforce the new time dimension
    ds = ds.assign_coords(time=times)
    # ds = ds.reindex(time=times, method='nearest')

    # load the dataset into memory
    logger.info('Loading dataset into memory...')
    ds.load()
    logger.info('Dataset loaded successfully.')

    # save dataset to output file
    logger.info(f'Saving output dataset: {cfg.output_dataset}')
    ds.to_netcdf(cfg.output_dataset, mode='w')
    logger.info('Supersampling completed successfully.')



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Super sample data to a new time dimension.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()
    main(args.config)