import os
import logging
import cdsapi
import omegaconf
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
ERA5 retrieval through the Copernicus Climate Data Store (cdsapi).

Drop-in alternative to era5_arco_retrieval.py: it consumes the same request
schema (see retrieval/configs/cds_retrieval_dev.yaml) and writes netcdf files
with the same layout (dimension `time`, data variable named `variable_name`,
no time dimension for constant fields). Requires a CDS account and a
~/.cdsapirc file.
"""

SINGLE_LEVELS = 'reanalysis-era5-single-levels'
PRESSURE_LEVELS = 'reanalysis-era5-pressure-levels'
CONSTANT_TIME = '1980-01-01T00:00:00'  # same convention as the arco utility


def _monthly_requests(variable_name: str, times: pd.DatetimeIndex, level: str = None):
    """Group times by (year, month) and build one CDS request for each group."""
    requests = []
    for (year, month), t in pd.Series(times, index=times).groupby([times.year, times.month]):
        t = pd.DatetimeIndex(t.values)
        params = {
            'product_type': ['reanalysis'],
            'variable': [variable_name],
            'year': [str(year)],
            'month': [f'{month:02d}'],
            'day': sorted({f'{d:02d}' for d in t.day}),
            'time': sorted({f'{h:02d}:00' for h in t.hour}),
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        }
        if level is not None:
            params['pressure_level'] = [str(int(float(level)))]
        requests.append((f'{year}{month:02d}', params))
    return requests


def retrieve_request(
        variable_name: str,
        times: np.ndarray,
        output_file: str,
        overwrite: bool = False,
        level: str = None,
        accumulation_period: int = None,
):
    """
    Download ERA5 reanalysis data from the Copernicus CDS at 0.25 degrees and
    save it to a netcdf file following the same conventions as
    era5_arco_retrieval.retrieve_request.
    Args:
        variable_name (str): CDS long variable name, e.g. '2m_temperature'.
        times (np.ndarray): Times to retrieve, datetime64.
        output_file (str): Path to the output file.
        overwrite (bool): If True, overwrite the output file if it already exists.
        level (str): Pressure level in hPa, e.g. '1000.'. None for single-level fields.
        accumulation_period (int): If specified, hourly accumulations are summed over
            this many hours (trailing window). Otherwise values are instantaneous.
    """
    if os.path.exists(output_file) and not overwrite:
        logger.info(f'{output_file} already exists. Skipping...')
        return
    if os.path.exists(output_file):
        logger.info(f'{output_file} already exists. Overwriting.')
        os.remove(output_file)

    times = pd.DatetimeIndex(times)
    constant = len(times) == 1

    # accumulation needs the preceding hourly steps for the trailing window
    if accumulation_period is not None:
        hourly = pd.date_range(times.min() - pd.Timedelta(hours=accumulation_period - 1),
                               times.max(), freq='1h')
        fetch_times = hourly
    else:
        fetch_times = times

    dataset = SINGLE_LEVELS if level is None else PRESSURE_LEVELS
    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)

    client = cdsapi.Client()
    temp_files = []
    for tag, params in _monthly_requests(variable_name, fetch_times, level):
        temp_file = output_file.replace('.nc', f'_{tag}.nc')
        temp_files.append(temp_file)
        if os.path.exists(temp_file):
            logger.info(f'Temporary file {temp_file} exists, skipping download...')
            continue
        logger.info(f'Requesting {dataset} {variable_name} for {tag}...')
        client.retrieve(dataset, params).download(temp_file)

    ds = xr.open_mfdataset(temp_files, combine='by_coords')
    if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    if 'pressure_level' in ds.coords:
        ds = ds.rename({'pressure_level': 'level'})
    # drop auxiliary coords (e.g. 'number', 'expver') and use the single data variable
    data_vars = [v for v in ds.data_vars if v not in ('number', 'expver')]
    if len(data_vars) != 1:
        raise ValueError(f'Expected one data variable in CDS response, got {data_vars}')
    da = ds[data_vars[0]].rename(variable_name)
    da = da.drop_vars([c for c in ('number', 'expver') if c in da.coords], errors='ignore')
    if 'level' in da.dims:
        da = da.squeeze('level', drop=False)

    if accumulation_period is not None:
        logger.info(f'Accumulating data over {accumulation_period} steps...')
        da = da.sortby('time').rolling(time=accumulation_period, min_periods=1).sum()
        da.attrs['accumulation_period'] = accumulation_period

    da = da.sel(time=times.to_numpy())
    if constant:
        logger.info('Only a single time selected. Assuming time-invariant field...')
        da = da.isel(time=0).drop_vars('time')

    logger.info(f'Writing final {output_file}...')
    with ProgressBar():
        da.to_netcdf(output_file)
    ds.close()
    for f in temp_files:
        os.remove(f)
    logger.info('Done.')


def main(config: str):
    cfg = omegaconf.OmegaConf.load(config)
    for request in cfg.requests:
        logger.info(f'Processing request: \n{omegaconf.OmegaConf.to_yaml(request)}')
        if request.get('constant', False):
            times = pd.to_datetime([CONSTANT_TIME]).to_numpy().astype('datetime64[ns]')
        else:
            times = pd.date_range(start=request.time_start, end=request.time_end,
                                  freq=request.time_freq).to_numpy().astype('datetime64[ns]')
        retrieve_request(
            variable_name=request.variable_name,
            times=times,
            output_file=request.output_file,
            accumulation_period=request.get('accumulation_period', None),
            level=request.get('level', None),
            overwrite=request.get('overwrite', False),
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ERA5 CDS retrieval script.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    main(config=args.config)
