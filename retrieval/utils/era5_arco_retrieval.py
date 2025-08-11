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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        times:np.array,
        partitions: int,
        output_file:str,
        overwrite:bool=False
):
    """
    Utility to download and process ERA5 reanalysis data from Google Cloud Storage.
    The data is interpolated to a 0.25 degree grid and saved to a netcdf file
    following DLWP data pipeline convenctions.
    Args:
        variable_name (str): Name of the variable to download, used for indexing
        times (np.array): Array of times to download. 
        partitions (int): Number of partitions to split the data into. This will 
            dictate memory resources used during processing, check what makes sense
            for your system.     
        output_file (str): Path to the output file. 
        overwrite (bool): If True, overwrite the output file if it already exists.
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
        # reanalysis = xr.open_zarr(
        #     'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3', 
        #     chunks={'time': 1},
        #     consolidated=True,
        # )
        ds = xr.open_zarr(
            'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
            chunks=None,
            storage_options=dict(token='anon'),
        )
        #.pipe(mirror_point_at_360)
        # fs = fsspec.filesystem('gs')
        # fs.ls('gs://gcp-public-data-arco-era5/co/')

        # # open dataset
        # logger.info('Openning dataset...')
        # reanalysis = xr.open_zarr(
        #     'gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr', 
        #     chunks={'time': 1},
        #     consolidated=True,
        # )#.pipe(mirror_point_at_360)

        print(f'ds variables: {ds.data_vars}')
        print(f'ds times: {ds.time}')
        print(f'first 5 times : {ds.time.values[:5]}')
        t = '2025-01-01T00:00:00.000000000'
        print(f'last timestep of the dataset: {t} ', ds["volumetric_soil_water_layer_1"].sel(time=t).values)
        exit()

        # interpolate to 0.25 degree grid
        logger.info('Interpolating to 0.25 degree grid...')
        tri = build_triangulation(reanalysis.longitude, reanalysis.latitude)
        longitude = np.linspace(0, 360, num=360*4+1)
        latitude = np.linspace(-90, 90, num=180*4+1)
        mesh = np.stack(np.meshgrid(longitude, latitude, indexing='ij'), axis=-1)

        # split times into partitions
        time_partitions = np.array_split(times, partitions)
        parition_filenames = [f'{output_file}.p{i}' for i in range(partitions)]
        pbar = tqdm(enumerate(time_partitions), total=partitions, ascii=True)
        for i, partition in pbar:
            pbar.set_description(f'Processing partition {i+1}/{partitions}...')

            # select variable and times 
            da = reanalysis[variable_name].sel(time=partition)
            da_mesh = interpolate(da.values, tri, mesh)
            da_ll = xr.DataArray(da_mesh, coords={'time': da.time.values, 'longitude': longitude, 'latitude': latitude}, dims=['time', 'longitude', 'latitude'])
            da_ll.name = variable_name

            # save partition file
            da_ll.to_netcdf(parition_filenames[i])

            # clean up
            del da, da_mesh, da_ll
        
        # merge partitions
        logger.info('Merging partitions...')
        da_ll = xr.open_mfdataset(parition_filenames, chunks={'time':1}, combine='by_coords', parallel=True)
        # reorder lattitude, follow dlwp pipeline convention  
        da_ll = da_ll.sel(latitude=slice(None, None, -1))
        # remove last, redundant longitude
        da_ll = da_ll.isel(longitude=slice(0, -1))
        
        # enforce dimesnions order: time, latitude, longitude
        da_ll = da_ll.transpose('time', 'latitude', 'longitude')
            
        # save to netcdf file
        logger.info(f'Writing final {output_file}...')
        with ProgressBar():
            da_ll.to_netcdf(output_file)
        logger.info('Done.')
        
        # clean up
        for f in parition_filenames:
            os.remove(f)

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
            times=pd.date_range(start=request.time_start, end=request.time_end, freq=request.time_freq),
            partitions=request.partitions,
            output_file=request.output_file,
            overwrite=request.overwrite
        )

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ERA5 retrieval script.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load the configuration
    main(config=args.config)