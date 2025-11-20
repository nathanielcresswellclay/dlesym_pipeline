import os
import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def constant_impute_ocean(
    filename: str,
    variable: str,
    constant_value: float,
    output_file: str,
    chunks: dict = {'time': 8},
    overwrite: bool = False,
    time_slice = None,
    output_plot_index: int = None,
):
    """
    Impute a constant value to a variable in a netCDF file.
    
    Args:
        filename (str): Path to the input netCDF file.
        variable (str): Name of the variable to impute.
        constant_value (float): Value to impute.
        output_file (str): Path to save the output netCDF file. If None, overwrites the input file.
        chunks (dict): Chunk sizes for the data loading.
        overwrite (bool): Whether to overwrite the output file if it exists.
        time_slice (slice): Optional time slice to select a subset of the data.
        output_plot_index (int): Optional index for output plot visualization.
    Returns:
        None
    """

    # check if output file exists
    if os.path.exists(output_file) and not overwrite:
        logger.info(f'Output file {output_file} exists and overwrite is False, skipping imputation...')
        return
    
    # load dataset, selecting time if specified
    ds = xr.open_dataset(filename, chunks=OmegaConf.to_container(chunks) if chunks != 'auto' else 'auto')[variable]
    # subsample if time_slice is specified
    if time_slice is not None:
        logger.info(f'Selecting time slice: {time_slice}')
        ds = ds.sel(time=slice(time_slice['start'], time_slice['end']))
    else:  
        logger.info(f'No time slice specified, using full time range: {ds.time.values[0]} to {ds.time.values[-1]}') 

    # impute
    ds = ds.fillna(constant_value)
    logger.info(f'Running imputation and saving output in {filename}')
    with ProgressBar():
        ds.to_netcdf(output_file, compute=True)
    logger.info(f'Imputation complete. Output saved to {output_file}')
    logger.info(ds)

    # plot single frame. for quick visualization during dev and debugging
    if output_plot_index is not None:
        ds
        logger.info(f'Generating output plot for index...')
        fig, ax = plt.subplots(figsize=(8,6))
        ds.isel(time=output_plot_index).plot(ax=ax)
        fig.savefig(output_file.replace('.nc', f'_imputed_variable_index_{output_plot_index}.png'))
        logger.info(f'Output plot saved to imputed_variable_index_{output_plot_index}.png')

def main(config: OmegaConf):

    for file_cfg in config: 

        logger.info(f'Imputing constant value over ocean. Config: {file_cfg}')
        constant_impute_ocean(
            filename=file_cfg['filename'],
            variable=file_cfg['variable'],
            constant_value=file_cfg['constant_value'],
            output_file=file_cfg['output_file'],
            chunks=file_cfg.get('chunks', 'auto'),
            overwrite=file_cfg.get('overwrite', False),
            time_slice=file_cfg.get('time_slice', None),
            output_plot_index=file_cfg.get('output_plot_index', None),
        )


if __name__=="__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Impute constant values over ocean ")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)
