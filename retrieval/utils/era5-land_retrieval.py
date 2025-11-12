import os
import numpy as np
import pandas as pd
import xarray as xr
import omegaconf
import logging
from dask.diagnostics import ProgressBar
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import cdsapi as cds

def _get_yearly_request(variable_name: str, year: int, time_freq: int):

    return {
        "variable": [variable_name],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(0, 24, time_freq)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": [0.25, 0.25],
    }

def main(config: omegaconf.DictConfig):


    # compile requests and intermediate filenames
    for request in config.requests:

        logger.info(f'Processing request: {request}')

        # check if output file exists
        if not os.path.exists(request.output_file) or request.overwrite:

            if os.path.exists(request.output_file):
                logger.info(f'Output file {request.output_file} exists and overwrite is True, overwriting...')
            # break in to years 
            api_params = []
            temp_files = []
            years = np.arange(request.start_year, request.end_year + 1)
            if len(years)>1:
                logger.info(f'Breaking request into yearly chunks: {years}')
                for year in years:
                    api_params.append(_get_yearly_request(request.variable_name, year, request.time_freq))
                    temp_files.append(f"{request.output_file.replace('.nc',f'_{year}.nc')}")
            else:
                api_params.append(_get_yearly_request(request.variable_name, years[0], request.time_freq))
                temp_files.append(request.output_file.replace('.nc',f'_{years[0]}.nc'))

            # run the requests
            for api_param, temp_file in zip(api_params, temp_files):
                
                client = cds.Client()
                client.retrieve("reanalysis-era5-land", api_param).download(temp_file)

            # combine yearly files if needed
            logger.info(f'Combining yearly files into final output: {request.output_file}')
            ds_list = [xr.open_dataset(f) for f in temp_files]
            ds_combined = xr.concat(ds_list, dim='valid_time')
            # rename time dimension
            ds_combined = ds_combined.rename({'valid_time': 'time'})
            logger.info(f'Chunking to time:{request.time_chunk_size} and writing final output to {request.output_file}...')
            ds_combined = ds_combined.chunk({'time': request.time_chunk_size})
            with ProgressBar():
                ds_combined.to_netcdf(request.output_file, mode='w')
            # close datasets
            for ds in ds_list:
                ds.close()
            ds_combined.close()
            # remove temporary files
            logger.info(f'Removing {len(temp_files)} temporary files...')
            for f in temp_files:
                os.remove(f)
            logger.info('Done.')
        else:
            logger.info(f'Output file {request.output_file} exists and overwrite is False, skipping...')

if __name__ == "__main__":

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run ERA5-land retrieval script.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # create config object
    config = omegaconf.OmegaConf.load(args.config)
    # Load the configuration
    main(config=config)