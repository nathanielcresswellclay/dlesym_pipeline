import omegaconf
import fsspec
import xarray as xr
import pandas as pd
import scipy.spatial
import numpy as np
import logging
import os
import dask.array as das
import dask
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import zarr
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mirror_point_at_360(ds):
  extra_point = (
      ds.where(ds.longitude == 0, drop=True)
      .assign_coords(longitude=lambda x: x.longitude + 360)
  )
  return xr.concat([ds, extra_point], dim='values')

def build_triangulation(x, y):
  grid = np.stack([x, y], axis=1)
  return scipy.spatial.Delaunay(grid)

def interpolate(data, tri, mesh):
  indices = tri.find_simplex(mesh)
  ndim = tri.transform.shape[-1]
  T_inv = tri.transform[indices, :ndim, :]
  r = tri.transform[indices, ndim, :]
  c = np.einsum('...ij,...j', T_inv, mesh - r)
  c = np.concatenate([c, 1 - c.sum(axis=-1, keepdims=True)], axis=-1)
  result = np.einsum('...i,...i', data[:, tri.simplices[indices]], c)
  return result #np.where(indices == -1, , result)

def retrieve_request(
        variable_name:str,
        times: np.ndarray,
        output_file:str,
        overwrite:bool=False,
        time_chunks: int = 1,
        level: str = None,
        accumulation_period: int = None,
):
    """
    Utility to download and process ERA5 reanalysis data from Google Cloud Storage.
    The data is interpolated to a 0.25 degree grid and saved to a netcdf file
    following DLWP data pipeline convenctions.
    Args:
        variable_name (str): Name of the variable to download, used for indexing
        times (np.ndarray): Array of times to retrieve data for, should be in pandas datetime format.   
        output_file (str): Path to the output file. 
        overwrite (bool): If True, overwrite the output file if it already exists.
        time_chunks (int): Number of time chunks to use for processing.
        level (str): Atmospheric level to retrieve data for, e.g. '1000.' for 1000 hPa.
        accumulation_period (int): If specified, the data will be accumulated over this period in hours. 
            otherwise data will be treated as instantaneous values.
    Returns:
        None: The function saves the processed data to the specified output file.
    """

    if os.path.exists(output_file) and not overwrite:
        logger.info(f'{output_file} already exists. Skipping...')
        return
    else:
        if os.path.exists(output_file):
            logger.info(f'{output_file} already exists. Overwriting.')
            os.remove(output_file)

        # mount cloud file system 
        logger.info('Mounting cloud file system...')
        fs = fsspec.filesystem('gs')
        fs.ls('gs://gcp-public-data-arco-era5/ar')

        # open dataset
        logger.info('Openning dataset...')
        ds = xr.open_zarr(
            'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
            chunks={'time': time_chunks},
            storage_options=dict(token='anon'),
        )[variable_name]

        if level is not None:
            # select the level
            logger.info(f'Selecting level {level}...')
            ds = ds.sel(level=level)
        
        # if accumulation period is specified, accumulate the data
        if accumulation_period is not None:
            logger.info(f'Accumulating data over {accumulation_period} steps...')
            ds = ds.rolling(time=accumulation_period, min_periods=1).sum()
            # add note in metadata
            ds.attrs['accumulation_period'] = accumulation_period

        # select the times
        constant = False
        if len(times)==1:
            logger.info(f'Only seleting a single time. Assuming time-invariant field...')
            constant = True
        ds = ds.sel(time=times)
        # remove time dimension if constant
        if constant:
            ds = ds.isel(time=0)
            ds = ds.drop('time')
            

        # enforce chunking of time dimension
        # ds = ds.chunk({'time': time_chunks})
        # save to netcdf file
        logger.info(f'Writing final {output_file}...')
        with ProgressBar():
            ds.to_netcdf(output_file)
        logger.info('Done.')
        # release memory
        ds.close()

    return

def main(config: str):
    """
    Main function to run the ERA5 retrieval script.
    Args:
        config (str): Path to the configuration file.
    """
    # Load the configuration
    cfg = omegaconf.OmegaConf.load(config)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    for request in cfg.requests:
        logger.info(f'Processing request: \n{omegaconf.OmegaConf.to_yaml(request)}')

        # Run the retrieval request
        retrieve_request(
            variable_name=request.variable_name,
            times=pd.date_range(start=request.time_start, end=request.time_end, freq=request.time_freq).to_numpy().astype('datetime64[ns]') \
                if not request.get('constant', False) else pd.to_datetime(['1980-01-01T00:00:00']).to_numpy().astype('datetime64[ns]'),
            output_file=request.output_file,
            accumulation_period=request.get('accumulation_period', None),
            level=request.get('level', None),
            overwrite=request.overwrite
        )

    

if __name__ == "__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run ERA5 retrieval script.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load the configuration
    main(config=args.config)