from typing import DefaultDict, Optional, Sequence, Union
from omegaconf import DictConfig, OmegaConf
import logging
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shutil
import time
from dask.diagnostics import ProgressBar
from pprint import pprint
import numpy as np
import pandas as pd
import xarray as xr
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_healpix(ax, data_hpx):
    """
    Plot HEALPix data (nside=64) on a regular 1° × 1° lat–lon grid.

    The input HEALPix field is remapped to latitude–longitude space using
    the project’s `HEALPixRemap` utility and displayed with `pcolormesh`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the plot.
    data_hpx : array-like
        HEALPix data compatible with `HEALPixRemap.hpx2ll`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The input Axes.
    im : matplotlib.collections.QuadMesh
        The object returned by `pcolormesh`.

    Raises
    ------
    ImportError
        If `HEALPixRemap` cannot be imported.
    """

    try:
        from processing.utils.healpix import HEALPixRemap
    except ImportError as e:
        logger.error("Failed to import reamp class. make sure you are running this utility from the project root directory (DLESyM/) and it's added to PYTHONPATH")
        raise ImportError(str(e))
    
    # initialize the remopper 
    mapper = HEALPixRemap(
            latitudes=181,
            longitudes=360,
            nside=64,
        )
    # map from hpx 
    data_ll = mapper.hpx2ll(data_hpx)
    lats = np.arange(90,-90.1,-1)
    lons = np.arange(0,360,1)
  
    # plot data 
    im = ax.pcolormesh(
        lons,
        lats,
        data_ll,
    )
    
    return ax, im

def plot_times_hpx(
        ds_path: str,
        times: list,
        output_dir: str,
):
    """
    Plot some sample times to check final dataset

    Args: 
        ds_path (str): path to open dataset 
        times (list): list of times to plot, interpretable by pandas (e.g. '2001-01-03T03:00:00')
        output_dir (str): output plots will be sent here. 
    """

    logger.info(f'Plotting samples times: {times} and saving to {output_dir}')

    # create output dir if not already existing
    os.makedirs(output_dir, exist_ok=True)

    # open_dataset 
    ds = xr.open_dataset(ds_path, engine='zarr')

    # loop through times and plot
    for t, time in enumerate(tqdm(times)): 

        # time string for file and title
        time_str = str(time)[:13]

        # get time
        ds_time = ds.sel(time=time)
        ds_time.load()

        ##############################################################################
        # INPUTS
        ##############################################################################

        # initialize input figure 
        fig_inputs, axs_inputs = plt.subplots(
            nrows=int(np.ceil(np.sqrt(len(ds_time.channel_in)))),
            ncols=int(np.ceil(np.sqrt(len(ds_time.channel_in)))),
            figsize=(7*np.ceil(np.sqrt(len(ds_time.channel_in))),
                     5*np.ceil(np.sqrt(len(ds_time.channel_in))),)
        )
        # plot inputs 
        for i, ax_channel in enumerate(axs_inputs.flatten()):
            ax_channel = axs_inputs.flatten()[i]
            try:
                c = ds_time.channel_in[i]
                ax_channel, im = plot_healpix(
                    ax = ax_channel,
                    data_hpx = ds_time.inputs.sel(channel_in=c).values,
                )
                fig_inputs.colorbar(im,ax=ax_channel)
                ax_channel.set_title(f'{c.values}', fontsize=15)
            except IndexError:
                # make extraneous plots invisible
                ax_channel.spines['top'].set_visible(False)
                ax_channel.spines['right'].set_visible(False)
                ax_channel.spines['bottom'].set_visible(False)
                ax_channel.spines['left'].set_visible(False)
                ax_channel.set_xticks([])
                ax_channel.set_yticks([])
        # title is date 
        fig_inputs.suptitle(f'Inputs: {time_str}',fontsize=20, y=1)
        # save input figure 
        fig_inputs.tight_layout()
        fig_inputs.savefig(f'{output_dir}/inputs_{str(ds_time.time.values)[:13]}.png',dpi=300)
        
        ##############################################################################
        # TARGETS
        ##############################################################################
        
        # initialize input figure 
        fig_targets, axs_targets = plt.subplots(
            nrows=int(np.ceil(np.sqrt(len(ds_time.channel_out)))),
            ncols=int(np.ceil(np.sqrt(len(ds_time.channel_out)))),
            figsize=(7*np.ceil(np.sqrt(len(ds_time.channel_out))),
                     5*np.ceil(np.sqrt(len(ds_time.channel_out))),)
        )
        # plot targets 
        for i, ax_channel in enumerate(axs_targets.flatten()):
            ax_channel = axs_targets.flatten()[i]
            try:
                c = ds_time.channel_out[i]
                ax_channel, im = plot_healpix(
                    ax = ax_channel,
                    data_hpx = ds_time.targets.sel(channel_out=c).values,
                )
                fig_targets.colorbar(im,ax=ax_channel)
                ax_channel.set_title(f'{c.values}', fontsize=15)
            except IndexError:
                # make extraneous plots invisible
                ax_channel.spines['top'].set_visible(False)
                ax_channel.spines['right'].set_visible(False)
                ax_channel.spines['bottom'].set_visible(False)
                ax_channel.spines['left'].set_visible(False)
                ax_channel.set_xticks([])
                ax_channel.set_yticks([])
        # title is date 
        fig_targets.suptitle(f'Targets: {time_str}',fontsize=20, y=1)
        # save input figure 
        fig_targets.tight_layout()
        fig_targets.savefig(f'{output_dir}/targets_{str(ds_time.time.values)[:13]}.png',dpi=300)

        ##############################################################################
        # CONSTANTS
        ##############################################################################
        if t == 0:
            # initialize input figure 
            fig_const, axs_const = plt.subplots(
                nrows=int(np.ceil(np.sqrt(len(ds_time.channel_c)))),
                ncols=int(np.ceil(np.sqrt(len(ds_time.channel_c)))),
                figsize=(7*np.ceil(np.sqrt(len(ds_time.channel_c))),
                        5*np.ceil(np.sqrt(len(ds_time.channel_c))),)
            )
            # plot const 
            for i, ax_channel in enumerate(axs_const.flatten()):
                ax_channel = axs_const.flatten()[i]
                try:
                    c = ds_time.channel_c[i]
                    ax_channel, im = plot_healpix(
                        ax = ax_channel,
                        data_hpx = ds_time.constants.sel(channel_c=c).values,
                    )
                    fig_const.colorbar(im,ax=ax_channel)
                    ax_channel.set_title(f'{c.values}', fontsize=15)
                except IndexError:
                    # make extraneous plots invisible
                    ax_channel.spines['top'].set_visible(False)
                    ax_channel.spines['right'].set_visible(False)
                    ax_channel.spines['bottom'].set_visible(False)
                    ax_channel.spines['left'].set_visible(False)
                    ax_channel.set_xticks([])
                    ax_channel.set_yticks([])
            # title is date 
            fig_const.suptitle(f'Const: {time_str}',fontsize=20, y=1)
            # save input figure 
            fig_const.tight_layout()
            fig_const.savefig(f'{output_dir}/const.png',dpi=300)
        
        # clear figures 
        plt.close(fig_inputs)
        plt.close(fig_targets)
        plt.close(fig_const)




def create_prebuilt_zarr(
        dst_directory: str,
        dataset_name: str,
        inputs: dict,
        outputs: dict,
        constants: dict,
        batch_size: int = 32,
        time_slice: slice = None,
        scaling: Optional[DictConfig] = None,
        time_dim: Sequence = None,
        overwrite: bool = False,
        ) -> xr.Dataset:
    """
    Create a prebuilt zarr dataset by merging multiple netcdf files into a single zarr file.
    Args:
        dst_directory (str): Directory where the zarr dataset will be saved.
        dataset_name (str): Name of the dataset to be created.
        inputs (dict): Dictionary of input variables and their corresponding file paths.
        outputs (dict): Dictionary of output variables and their corresponding file paths.
        constants (Optional[DefaultDict]): Dictionary of constants to be included in the dataset.
        batch_size (int): Size of the batches for chunking the dataset.
        time_slice (slice): Slice object to subset the time dimension.
        scaling (Optional[DictConfig]): Dictionary containing scaling parameters for the variables.
        time_dim (Sequence): Sequence of time values to ensure the time dimension is as desired.
        overwrite (bool): If True, overwrite the existing dataset if it exists.
    """

    # check if output file exists already 
    file_exists = os.path.exists(os.path.join(dst_directory, dataset_name + ".zarr"))

    # remove if overwrite and exists, return if exists and not overwrite
    if file_exists and not overwrite:
        logger.info(f"Dataset {dataset_name} already exists in {dst_directory}. To overwrite, set 'overwrite' to True. Aborting zarr creation.")
        return
    elif file_exists and overwrite:
        logger.info(f"Dataset {dataset_name} already exists in {dst_directory}. Overwriting.")
        shutil.rmtree(os.path.join(dst_directory, dataset_name + ".zarr"))

    # compile comprehensive list of variables and files (for looping)
    input_variables = list(inputs.keys())
    output_variables = list(outputs.keys()) or input_variables
    all_variables = np.union1d(input_variables , output_variables)
    merged_dict = {**inputs, **outputs}

    # log for visibility
    logger.info('Creating a zarr dataset by merging the following netcdf files:')
    logger.info(merged_dict)
    if time_slice is not None:
        logger.info(f"Time slice: {time_slice}")

    # for timing info later 
    merge_time = time.time()

    # loop through variable files and standardize 
    datasets = []
    remove_attrs = ['varlev', 'mean', 'std']
    for variable in all_variables:
        file_name = merged_dict[variable]
        # correct old convention if present, load lazily with batches
        if "sample" in list(xr.open_dataset(file_name).dims.keys()):
            ds = xr.open_dataset(file_name, chunks={'sample': batch_size}).rename({"sample": "time"})
        else:
            ds = xr.open_dataset(file_name, chunks={"time": batch_size})
        # agan accomodating old conventions. We only have one "varlev" per file
        if "varlev" in ds.dims:
            ds = ds.isel(varlev=0)
        # standardize time dimension
        if time_slice is not None:
            ds = ds.sel(time=time_slice)
        for attr in remove_attrs:
            try:
                ds = ds.drop(attr)
            except ValueError:
                pass
        # Rename variable
        if "predictors" in list(ds.keys()):
            ds = ds.rename({"predictors": variable})

        # Change lat/lon to coordinates
        try:
            ds = ds.set_coords(['lat', 'lon'])
        except (ValueError, KeyError):
            pass

        # Apply log scaling lazily
        if scaling is not None:
            if variable in scaling and scaling[variable].get('log_epsilon', None) is not None:
                ds[variable] = np.log(ds[variable] + scaling[variable]['log_epsilon']) \
                            - np.log(scaling[variable]['log_epsilon'])
            
        # check time dimension is as desired and if not, resample 
        if time_dim is not None:
            if not np.array_equal(ds.time.values.astype('datetime64[h]'),time_dim.astype('datetime64[h]')):
                logger.info(f'Time dimension of {merged_dict[variable]} is {ds.time.values.astype("datetime64[h]")}')
                logger.info(f' This is different from the specified time_dim. Resampling using forward fill to {time_dim.astype("datetime64[h]")}')
                ds = ds.reindex(time=time_dim, method='ffill')
        datasets.append(ds)

    # Merge datasets
    data = xr.merge(datasets, compat="override")

    # Convert to input/target array by merging along the variables
    input_da = data[list(input_variables)].to_array('channel_in', name='inputs').transpose(
        'time', 'channel_in', 'face', 'height', 'width')
    target_da = data[list(output_variables)].to_array('channel_out', name='targets').transpose(
        'time', 'channel_out', 'face', 'height', 'width')

    result = xr.Dataset()
    result['inputs'] = input_da
    result['targets'] = target_da

    # Get constants
    constants_ds = []
    for name, filename in constants.items():
        ds_const_temp = xr.open_dataset(filename)
        # add lat and lon to coordinates if included in the ds
        if 'lat' in ds_const_temp.data_vars and 'lon' in ds_const_temp.data_vars:
            ds_const_temp = ds_const_temp.set_coords(['lat','lon'])
        constants_ds.append(ds_const_temp[name].astype(np.float32))
    constants_ds = xr.merge(constants_ds, compat='override')
    constants_da = constants_ds.to_array('channel_c', name='constants').transpose(
        'channel_c', 'face', 'height', 'width')
    result['constants'] = constants_da

    # writing out
    def write_zarr(data, path):
        #write_job = data.to_netcdf(path, compute=False)
        write_job = data.to_zarr(path, encoding={'time':{'dtype':'float64'}}, compute=False) # we have to enforce float64 for time to avoid percision issues with zarr writing
        with ProgressBar():
            logger.info(f"writing zarr dataset to {path}")
            write_job.compute()
        logger.info('Successfully wrote zarr:')
        logger.info(data)

    # enforce chunking before writing
    result = result.chunk({'time': batch_size})

    write_zarr(data=result, path=os.path.join(dst_directory, dataset_name + ".zarr"))
    
    return True

def main(config:str):
    """
    Main function to remap a netCDF file to HEALPix format using the HEALPixRemap class.
    
    Parameters:
        config (str): Path to the configuration file containing parameters for remapping.
    Returns:
        None: The function performs the remapping and saves the output file.
    """

    # load config from yaml
    params = OmegaConf.load(config)

    # resolve time dimension
    time_dim = pd.date_range(
        params.time_dim.start,
        params.time_dim.end,
        freq=params.time_dim.freq,
    ).to_numpy()
    # get varaible identifier within otuput file name, this should only be different from target_variable_name for topography
    create_prebuilt_zarr(
        dst_directory=params.dst_directory,
        dataset_name=params.dataset_name,
        inputs=params.inputs,
        outputs=params.outputs,
        constants=params.constants,
        batch_size=getattr(params, 'batch_size', 32),
        time_slice=slice(params.time_slice.start,params.time_slice.end),
        scaling=getattr(params, 'scaling', None),
        time_dim=time_dim,
        overwrite=getattr(params, 'overwrite', False)
    )

    # option to specify times for plotting inputs. Sanity check
    if 'plotting_times' in params.keys():
        plot_times_hpx(
            ds_path = os.path.join(params.dst_directory,f'{params.dataset_name}.zarr'),
            times = params.plotting_times,
            output_dir = params.plotting_output_dir,
        )

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Compile HPX64 data into a zarr dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    main(config=args.config)