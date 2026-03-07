import sys 
import argparse
import sys
import os
import numpy as np
import omegaconf
import xarray as xr
import healpy as hp
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from processing.utils.healpix import HEALPixRemap
from compilation.utils.write_zarr import plot_healpix

def _remap(
        file_name,
        target_variable_name,
        file_variable_name,
        target_file_variable_name=None,
        prefix='',
        nside=64,
        order='bilinear',
        resolution_factor=1.0,
        visualize=False,
        times=None,
        pool_size=1,
        overwrite=False,
        plotting=None,
):  

    """
    This function remaps a latlon dataset to a HEALPix dataset.
    
    Parameters:
    params (dict): A dictionary with the following keys:
        - 'file_name': including absolute path of lat lon file to be remapped.
        - 'target_variable_name': variable name to be used in the output dataset.
        - 'file_variable_name': name of variable inside input file.
        - 'target_file_variable_name': name of variable used to create output file name, only if different from 'target_variable_name'.
        - 'prefix': path and file prefix for output file name (to be combined with file_variable_name).
        - 'nside': HEALPix nside parameter e.g. 32.
        - 'order': interpolation order e.g. 'bilinear'.
        - 'resolution_factor': resolution factor for remap (adjust with caution) e.g. 1.0.
        - 'visualize': boolean to visualize remap (warning: buggy).
        - 'times': list of time indices to remap (defaults to all times).
        - 'pool_size': number of parallel processes to use (defaults to 1).
        - 'overwrite': boolean to overwrite existing files (defaults to False).
        - 'plotting': dictionary with the following keys:
            - 'dir': directory to save plotting frames
            - 'times': list of time indices to plot
    """

    # make sure target file doesn't already exits 
    logger.info('checking parameters...')
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f'input file ~{file_name}~ does not exist. Aborting.')
    if target_file_variable_name is None:
        target_file_variable_name = target_variable_name
    if os.path.isfile(prefix+target_file_variable_name+'.nc'):
        if overwrite:
            logger.info(f'target file ~{prefix+target_file_variable_name+".nc"}~ already exists. Overwriting as requested.')
        else:
            logger.info(f'target file ~{prefix+target_file_variable_name+".nc"}~ already exists. Aborting.')
            return
    
    # resolve times to remap
    if times is not None:
        times = slice(times['start'], times['end'])
        logger.info(f'remapping times from {times.start} to {times.stop}...')

    # make directory of output file if it doesn't exist
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    assert f"HPX{nside}" in prefix, (f"'HPX{nside}' could not be found in the prefix '{prefix}'. Please make sure "
                                    f"that the nside={nside} parameter and the prefix provided to this function "
                                    f"match.")

    # Load .nc file in latlon format to extract latlon information and to initialize the remapper module
    logger.info(f'loading latlon dataset {file_name}...')
    ds_ll = xr.open_dataset(file_name)
    logger.info(f'loaded latlon dataset {ds_ll}')
    if 'time' in xr.open_dataset(file_name).dims:
        ds_ll.rename({"time": "sample"}).squeeze()
    try:
        latitudes, longitudes = ds_ll.dims["latitude"], ds_ll.dims["longitude"]
    except KeyError:
        latitudes, longitudes = ds_ll.dims["lat"], ds_ll.dims["lon"]
    mapper = HEALPixRemap(
        latitudes=latitudes,
        longitudes=longitudes,
        nside=nside,
        resolution_factor=resolution_factor,
        order=order,
        )
    
    # release_memory
    ds_ll.close()
    # determine weather remap is to be done in parallel 
    output_file = prefix + target_file_variable_name + '.nc'
    mapper.remap(
        file_path=file_name,
        prefix=prefix,
        file_variable_name=file_variable_name,
        target_variable_name=target_variable_name,
        target_file_variable_name=target_file_variable_name,
        poolsize=pool_size,
        chunk_ds=True,
        times=times,
        output_file=output_file,
    )

    if plotting is not None:
        logger.info(f'plotting sanity-check frames: {plotting}')
        output_dir = plotting['dir']
        # make plotting dir
        os.makedirs(output_dir, exist_ok=True)
        # plot times: HEALPix remapped + source lat-lon side by side
        ds = xr.open_dataset(output_file)
        ds_src = xr.open_dataset(file_name)
        # coordinate names in source (lat/lon or latitude/longitude)
        lat_dim, lon_dim = ('latitude', 'longitude') if 'latitude' in ds_src.dims else ('lat', 'lon')
        lat_coord = ds_src[lat_dim].values
        lon_coord = ds_src[lon_dim].values
        for time in tqdm(plotting['times']):
            time_alias = time[:13]
            ds_time = ds.sel(time=time)
            # plot_healpix expects a numpy array (12, nside, nside), not an xarray Dataset
            data_hpx = np.squeeze(ds_time[target_variable_name].values)
            data_ll = ds_src.sel(time=time)[file_variable_name]
            data_ll_values = np.squeeze(data_ll.values)
            # shared color scale for comparison
            vmin = min(np.nanmin(data_hpx), np.nanmin(data_ll_values))
            vmax = max(np.nanmax(data_hpx), np.nanmax(data_ll_values))
            fig, (ax_hpx, ax_ll) = plt.subplots(1, 2, figsize=(14, 5))
            ax_hpx, im_hpx = plot_healpix(ax_hpx, data_hpx)
            im_hpx.set_clim(vmin, vmax)
            ax_hpx.set_title('HEALPix (remapped)')
            fig.colorbar(im_hpx, ax=ax_hpx)
            im_ll = ax_ll.pcolormesh(lon_coord, lat_coord, data_ll_values, vmin=vmin, vmax=vmax)
            ax_ll.set_title('Source (lat–lon)')
            ax_ll.set_xlabel(lon_dim)
            ax_ll.set_ylabel(lat_dim)
            fig.colorbar(im_ll, ax=ax_ll)
            fig.suptitle(f'{target_variable_name} — {time_alias}')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{time_alias}.png'))
            plt.close(fig)
        ds.close()
        ds_src.close()

def main(config):
    """
    Main function to run the remapping process.
    
    Parameters:
    config (str): Path to the configuration file containing the parameters for remapping.
    """
    
    # Load the configuration file
    cfg = omegaconf.OmegaConf.load(config)

    # if is a list of configs, run each one
    for single_cfg in cfg:
        _remap(
            file_name=single_cfg.file_name,
            target_variable_name=single_cfg.target_variable_name,
            file_variable_name=single_cfg.file_variable_name,
            target_file_variable_name=single_cfg.get('target_file_variable_name', None),
            prefix=single_cfg.prefix,
            nside=single_cfg.nside,
            order=single_cfg.order,
            resolution_factor=single_cfg.resolution_factor,
            visualize=single_cfg.visualize,
            times=single_cfg.get('times', None),
            pool_size=single_cfg.get('pool_size', 1),
            overwrite=single_cfg.get('overwrite', False),
            plotting=single_cfg.get('plotting', None),
        )

if __name__ == "__main__":
     
    import argparse
    parser = argparse.ArgumentParser(description="Run remapping")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load the configuration
    main(config=args.config)
