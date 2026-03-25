"""
Transform ttr1h to olr by scaling ttr to match mean and standard deviation of olr.

This script is driven by an OmegaConf/YAML configuration file (see
`processing/configs/transform_ttr1h-olr_dev.yaml` for an example).

It:
  1. Opens ttr1h NetCDF file as xarray dataset.
  2. Opens olr NetCDF file as xarray dataset.
  3. Calculate mean and standard deviation of ttr1h of olr during the days of the year contained in the ttr1h file.
  4. Scales ttr1h to match mean and standard deviation of olr.
  5. Writes the result to olr NetCDF file.
"""

import os
import time as time_utils
import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from compilation.utils.write_zarr import plot_healpix

def main(config):
    """
    Parameters
    ----------
    config : omegaconf.DictConfig
        Expected fields:
          - `ttr_filename` (str): path to a NetCDF file containing variable `ttr1h`
          - `olr_filename` (str): path to a NetCDF file containing variable `olr`
          - `overwrite` (bool): whether to recompute if `olr_filename` exists
          - `output_filename` (str): NetCDF file output path
          - `constraints` (dict, optional): post-scaling value constraints
            - `min_quantile` (float): floor at this OLR quantile (0..1)
            - `global_floor` (bool): if True, use a scalar floor; if False, per-gridcell floor over time
          - `plotting` (dict): dictionary containing the plotting configuration
            - `dir` (str): directory where the plots will be saved
            - `times` (list[str]): list of timestamps to plot
    """

    if not os.path.exists(config.output_filename) or config.overwrite:

        logger.info(f"Opening data from {config.ttr_filename} and {config.olr_filename}...")

        # resolve chunks 
        chunks = OmegaConf.to_container(config.chunks)
        # Open ttr1h and olr component data
        ds_ttr = xr.open_dataset(config.ttr_filename,chunks=chunks if not config.in_memory else None)['ttr1h']
        ds_olr = xr.open_dataset(config.olr_filename,chunks=chunks if not config.in_memory else None)['olr']

        if config.in_memory:
            logger.info("Loading datasets into memory...")
            ds_ttr = ds_ttr.load()
            ds_olr = ds_olr.load()
            logger.info("...loaded datasets into memory.")

        if 'sample' in ds_olr.dims:
            ds_olr = ds_olr.rename({'sample': 'time'})

        # Determine overlap in day-of-year between the two time series.
        # We match mean/std computed over TIME for those overlapping day-of-year values.
        ttr_doy = ds_ttr["time"].dt.dayofyear
        olr_doy = ds_olr["time"].dt.dayofyear
        overlap_doy = np.intersect1d(
            np.unique(ttr_doy.values),
            np.unique(olr_doy.values),
        )
        if overlap_doy.size == 0:
            raise ValueError(
                "No overlapping day-of-year values between ttr and olr time coordinates."
            )

        ds_ttr_overlap = ds_ttr.where(ttr_doy.isin(overlap_doy), drop=True)
        ds_olr_overlap = ds_olr.where(olr_doy.isin(overlap_doy), drop=True)

        # Compute target/source statistics over time (leaving spatial dims intact).
        olr_mean = ds_olr_overlap.mean(dim="time")
        olr_std = ds_olr_overlap.std(dim="time")
        ttr1h_mean = ds_ttr_overlap.mean(dim="time")
        ttr1h_std = ds_ttr_overlap.std(dim="time")

        # first remove mean and std from ttr1h and invert the sign
        ttr1h_scaled = ((ds_ttr - ttr1h_mean) / ttr1h_std) * -1 

        # then scale to match mean and standard deviation of olr
        ttr1h_scaled = ttr1h_scaled * olr_std + olr_mean

        # Optional post-transform lower-bound constraint:
        # use source OLR climatology to prevent non-physical low tails.
        if "constraints" in config and config.constraints is not None:
            min_quantile = config.constraints.get("min_quantile", None)
            if min_quantile is not None:
                if not (0.0 <= float(min_quantile) <= 1.0):
                    raise ValueError("constraints.min_quantile must be in [0, 1].")
                use_global_floor = bool(config.constraints.get("global_floor", False))
                if use_global_floor:
                    floor_value = ds_olr_overlap.quantile(min_quantile)
                else:
                    floor_value = ds_olr_overlap.quantile(min_quantile, dim="time")
                logger.info(
                    "Applying lower-bound floor from source OLR quantile q=%.3f (global_floor=%s).",
                    float(min_quantile),
                    use_global_floor,
                )
                ttr1h_scaled = xr.where(ttr1h_scaled < floor_value, floor_value, ttr1h_scaled)

        # save scaled ttr1h to netCDF file
        olr_result = ttr1h_scaled.rename('olr')
        logger.info(f'Saving olr to {config.output_filename}...')
        # For small time windows, dask scheduler overhead can dominate write time.
        # Materialize in-memory first to reduce many tiny write tasks.
        write_start = time_utils.perf_counter()
        os.makedirs(os.path.dirname(config.output_filename), exist_ok=True)
        if config.in_memory:
            logger.info("Writing in-memory...")
            olr_result.to_netcdf(config.output_filename, mode='w')
        else:
            logger.info("Writing in chunks...")
            olr_result = olr_result.chunk(config.chunks)
            with ProgressBar():
                olr_result.to_netcdf(config.output_filename, mode='w')
            logger.info("...Done!")
        logger.info("Write finished in %.2f seconds.", time_utils.perf_counter() - write_start)
        logger.info('...Done!')
    else:
        logger.info(f"File {config.output_filename} already exists and overwrite is set to False. Skipping computation.")

    # Plot frames for timestamps specified in config
    os.makedirs(config.plotting.dir, exist_ok=True)
    if config.plotting is not None:
        logger.info(f'plotting sanity-check frames: {config.plotting}')
        # plot times: HEALPix remapped + source lat-lon side by side
        ds = xr.open_dataset(config.output_filename)
        for ts in tqdm(config.plotting['target_times']):
            time_alias = ts[:13]
            ds_time = ds.sel(time=ts)
            # plot_healpix expects a numpy array (12, nside, nside), not an xarray Dataset
            data_hpx = np.squeeze(ds_time['olr'].values)
            # shared color scale for comparison
            vmin = config.plotting.vmin
            vmax = config.plotting.vmax
            fig, ax_hpx = plt.subplots(figsize=(5, 5))
            ax_hpx, im_hpx = plot_healpix(ax_hpx, data_hpx)
            im_hpx.set_clim(vmin, vmax)
            ax_hpx.set_title('olr')
            fig.colorbar(im_hpx, ax=ax_hpx)
            fig.suptitle(f'Result: olr — {time_alias}, min: {np.nanmin(data_hpx)}, max: {np.nanmax(data_hpx)}')
            fig.tight_layout()
            fig.savefig(os.path.join(config.plotting.dir, f'{time_alias}-result.png'))
            plt.close(fig)
        ds.close()
        ds_source = xr.open_dataset(config.olr_filename)
        if 'sample' in ds_source.dims:
            ds_source = ds_source.rename({'sample': 'time'})
        for ts in tqdm(config.plotting['source_times']):
            time_alias = ts[:13]
            ds_time = ds_source.sel(time=ts)
            data_hpx = np.squeeze(ds_time['olr'].values)
            vmin = config.plotting.vmin
            vmax = config.plotting.vmax
            fig, ax_hpx = plt.subplots(figsize=(5, 5))
            ax_hpx, im_hpx = plot_healpix(ax_hpx, data_hpx)
            im_hpx.set_clim(vmin, vmax)
            ax_hpx.set_title('olr')
            fig.colorbar(im_hpx, ax=ax_hpx)
            fig.suptitle(f'Source: olr — {time_alias}, min: {np.nanmin(data_hpx)}, max: {np.nanmax(data_hpx)}')
            fig.tight_layout()
            fig.savefig(os.path.join(config.plotting.dir, f'{time_alias}-source.png'))
            plt.close(fig)
        ds_source.close()

        

    return

if __name__=="__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Transform ttr1h to olr by scaling ttr to match mean and standard deviation of olr.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)

