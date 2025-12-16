import xarray as xr
import pandas as pd
from omegaconf import OmegaConf
import numpy as np
import os
from dask.diagnostics import ProgressBar
import os
import numpy as np 
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def trailing_average(
    filename: str, 
    variable_name: str, 
    output_variable_name: str, 
    ds_dt: np.timedelta64,
    output_filename: str,
    influence_window: np.timedelta64,
    times: slice = None,
    chunks: dict = {'time': 64},
    load_first: bool = False,
    overwrite: bool = False):

    # create output directory if it does not exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # check if output file already exists
    if os.path.isfile(output_filename):
        if overwrite:
            logger.warning(f'Trailing Average: Target file {output_filename} already exists. Overwriting as requested.')
        else:
            logger.info(f'Trailing Average: Target file {output_filename} already exists. Aborting.')
            return

    # Open dataset lazily with Dask
    ds = xr.open_dataset(filename, chunks=chunks)
    da = ds[variable_name]

    # apply time slice if provided
    if times is not None:
        logger.info(f'Applying time slice: {times}')
        da = da.sel(time=times)

    # Apply rolling mean instead of manual looping
    window_size = int(influence_window / ds_dt)  # Convert timedelta to integer
    
    try:
        result = da.rolling(sample=window_size, center=False).mean()
    except KeyError:
        print(f'key error in rolling average, using time instead of sample for rolling average dim...')
        result = da.rolling(time=window_size, center=False).mean()

    # enforce chunks 
    result = result.chunk(chunks)
    # Rename variable
    result = result.rename(output_variable_name)

    logger.info(f'Writing trailing averaged data to {output_filename}...')
    with ProgressBar():
        result.to_netcdf(output_filename, compute=True)  # Compute only once at write stage

    logger.info('finished.')
    ds.close()

def main(config):

    # Load configparameters from configuration file
    trailing_average_param_list = OmegaConf.load(config)

    # loop though parameters and run trailing average calculation
    for params in trailing_average_param_list:
        trailing_average(
            filename=params.filename,
            variable_name=params.variable_name,
            output_variable_name=params.output_variable_name,
            ds_dt=pd.Timedelta(params.ds_dt).to_numpy(),
            output_filename=params.output_filename,
            influence_window=pd.Timedelta(params.influence_window).to_numpy(),
            times=slice(params.times.start, params.times.end) if params.get("times", None) is not None else None,
            chunks=OmegaConf.to_container(params.chunks),
            load_first=params.get("load_first", False),
            overwrite=params.get("overwrite", False)
        )
    


if __name__ == "__main__":
     
    import argparse
    parser = argparse.ArgumentParser(description="Run trailing average processing.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load the configuration
    main(config=args.config)
