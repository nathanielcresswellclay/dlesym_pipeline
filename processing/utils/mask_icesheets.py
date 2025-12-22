import xarray as xr
import pandas as pd
from omegaconf import OmegaConf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import os
import numpy as np 
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import cartopy.io.shapereader as shp
from shapely.geometry import Point

from compilation.utils.write_zarr import plot_healpix

def _get_mask(lat, lon, mask_lat_south=False):
    """
    Return spatial mask with true over Greenland and Antarctica, false elsewhere

    Params:
      lat: array of latitudes associated with each healpix point. Shape [12,64,64]
      lon: array of longitudes associated with each healpix point. Shape [12,64,64]
      mask_lat_south: ERA5-land has spurious values over anarctic sea ice. If true
         all regions south of 60S will be masked.
    Returns:
      mask: numpy array of boolean mask. Shape [12,64,64]
    """

    logger.info(f'Creating mask for Greenland and Antarctica...')
    reader = shp.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )
    records = list(shp.Reader(reader).records())

    GREENLAND = next(r.geometry for r in records if r.attributes["NAME"] == "Greenland")
    ANTARCTICA = next(r.geometry for r in records if r.attributes["NAME"] == "Antarctica")

    mask = np.zeros(lat.shape, dtype=bool)

    # iterate over all points
    it = np.nditer(lat, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        la = lat[idx]
        lo = lon[idx]

        # optional: normalize longitude to [-180, 180]
        if lo > 180:
            lo = lo - 360
        p = Point(lo, la)

        if GREENLAND.contains(p) or ANTARCTICA.contains(p):
            mask[idx] = True

        # mask 
        if mask_lat_south:
            if la < -60:
                mask[idx] = True

        it.iternext()

    logger.info('...finished')

    return mask

def mask_icesheet(
        filename: str,
        varname: str,
        fill_value: str, 
        output_filename: str,
        times: slice = None,
        chunks = 'auto',
        in_memory: bool = False,
        overwrite: bool = False,
        mask_antarctic: bool = False,
        plotting: dict = None,
):
    # open dataset
    ds = xr.open_dataset(filename, chunks=chunks)
    # get lat/lon encoding 
    lat = xr.open_dataset(filename, chunks=chunks).lat.values
    lon = xr.open_dataset(filename, chunks=chunks).lon.values
    # create icesheet mask
    mask = _get_mask(lat,lon,mask_antarctic)

    # subset if indicated 
    if times is not None: 
        logger.info(f'subsetting to slice: {times}')
        ds = ds.sel(time=times)

    # check overwrite indication
    if os.path.isfile(output_filename) and not overwrite:
        logger.info(f'file {output_filename} already exists and overwrite is {overwrite}. Skipping...')
    else:

        # build DataArray mask with correct dims
        mask_da = xr.DataArray(
            mask,
            dims=ds.lat.dims,
            coords={d: ds.coords[d] for d in ds.lat.dims},
        )
        # apply mask: replace True locations with fill_value
        ds_mask = ds.where(~mask_da, other=fill_value)

        # save file. Loaded into memory first or otherwise.
        os.makedirs(os.path.dirname(output_filename), exist_ok=True) # make dir first, if necessary
        if in_memory:
            logger.info('attempting to load dataset into memory...')
            ds_mask.load()
            # save 
            logger.info(f'saving masked file to {output_filename}')
            ds_mask.to_netcdf(output_filename, mode='w')
        else: 
            logger.info(f'saving masked file to {output_filename} in chunks')
            with ProgressBar():
                ds_mask.to_netcdf(output_filename, mode='w')
        logger.info('...done')

    if plotting is not None:
        logger.info(f'plotting sanity-check frames: {plotting}')
        output_dir = plotting['dir']
        # first plot mask
        fig,ax = plt.subplots(figsize=(10,5))
        ax, im = plot_healpix(ax, mask)
        fig.savefig(f'{output_dir}/icesheet_mask_{varname}.png',dpi=200)
        fig.colorbar(im,ax=ax, label='icesheet mask')
        plt.close(fig)

        # open original and new datasets 
        ds = ds = xr.open_dataset(filename)[varname]
        ds_mask = xr.open_dataset(output_filename)[varname]

        for time in tqdm(plotting['times']):
            time_alias = time[:13]
            # select data before and after masking 
            ds_time = ds.sel(time=time)
            ds_time_mask = ds_mask.sel(time=time)
            # Now plot time for original field and masked
            fig,axs = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
            axs[0], im = plot_healpix(axs[0],ds_time)
            fig.colorbar(im, ax=axs[0],label=varname)
            axs[0].set_title(f'Before Icesheet Mask: {time}')
            axs[1], im_mask = plot_healpix(axs[1],ds_time_mask)
            fig.colorbar(im_mask, ax=axs[1],label=varname)
            axs[1].set_title(f'After Icesheet Mask: {time}')
            # format output filename and save 
            plot_file = f"{plotting['dir']}/icesheet_masking_{varname}_{time_alias}.png"
            fig.savefig(plot_file,dpi=200)
            plt.close(fig)

def main(config):

    # Load configparameters from configuration file
    param_list = OmegaConf.load(config)

    # loop through file params and run ice sheet mask 
    for param in param_list:

        # log config 
        logger.info(f'Running icesheet mask for {param}')
        # send to mask
        mask = mask_icesheet(
            filename=param.filename,
            varname=param.varname,
            fill_value=param.fill_value,
            output_filename=param.output_filename,
            times=slice(param.times.start,param.times.end),
            chunks=param.chunks,
            overwrite=getattr(param,'overwrite',False),
            mask_antarctic=getattr(param,'mask_antarctic',False),
            plotting=getattr(param,'plotting',None),
        )

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Mask Greenland and Antarctiv icesheets")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    main(config=args.config)