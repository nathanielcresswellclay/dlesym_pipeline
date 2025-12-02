import omegaconf
import argparse
import logging
import os
import earthaccess
import pandas as pd
import yaml
import pathlib
from datetime import datetime, timedelta
from time import sleep

import re
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_netrc(creds_path: str):
    """Write Earthdata credentials to ~/.netrc"""
    with open(creds_path, "r") as f:
        creds = yaml.safe_load(f)
    username = creds.get("username")
    password = creds.get("token")
    if not username or not password:
        raise ValueError("Credentials YAML must contain 'username' and 'token'.")

    netrc_path = pathlib.Path.home() / ".netrc"
    content = f"""machine urs.earthdata.nasa.gov
login {username}
password {password}
"""
    netrc_path.write_text(content)
    netrc_path.chmod(0o600)  # secure permissions
    logger.info(f"Written Earthdata credentials to {netrc_path}")


def chunk_date_range(start_date: str, end_date: str, chunk_days: int = 30):
    """Split start/end dates into chunks of chunk_days"""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current.date().isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)

def download_maiac_ndvi_chunk(date_range, out_dir, max_retries=5):
    """Download a single chunk of MAIAC NDVI"""
    download_counter = 0
    failed_downloads = 0
    for attempt in range(1, max_retries + 1):
        try:
            results = earthaccess.search_data(
                short_name="MCD19A3CMG",
                bounding_box=(-180.0, -90.0, 180.0, 90.0),  # global
                temporal=date_range,
            )

            if not results:
                logger.warning(f"No granules found for {date_range}")
                return download_counter, failed_downloads

            logger.info(f"Granules found: {len(results)}")
            earthaccess.download(results, out_dir)
            download_counter += len(results)
            break  # success, break retry loop
        except Exception as e:
            logger.error(f"Attempt {attempt} failed for {date_range}: {e}")
            if attempt < max_retries:
                sleep(5 * attempt)  # exponential backoff
            else:
                logger.error(f"Failed to download {date_range} after {max_retries} attempts.")
                failed_downloads += len(results)

    return download_counter, failed_downloads



def download_maiac_ndvi(date_range, out_dir, chunk_days):
    """Download MAIAC NDVI over a long date range in chunks"""
    logger.info(f'Starting MAIAC download from {date_range[0]} to {date_range[1]} into {out_dir} in chunks of {chunk_days} days.')
    download_counter = 0
    failed_downloads = 0
    for chunk_start, chunk_end in chunk_date_range(date_range[0], date_range[1], chunk_days):
        logger.info(f"Processing chunk {chunk_start} to {chunk_end}")
        dc, fd = download_maiac_ndvi_chunk((chunk_start, chunk_end), out_dir)
        download_counter += dc
        failed_downloads += fd

    # check download size and log summary
    num_expected_files = len(pd.date_range(start=date_range[0], end=date_range[1], freq='D'))
    logger.info(f"Total downloaded granules: {download_counter}. Expected: {num_expected_files}")
    if failed_downloads > 0:
        logger.warning(f"Total failed granules: {failed_downloads}")

def main(config: omegaconf.DictConfig):
    creds_path = config.get("credentials")
    if creds_path:
        setup_netrc(creds_path)

    earthaccess.login()  # uses ~/.netrc

    download_maiac_ndvi(
        date_range=(config.start_date, config.end_date),
        out_dir=config.output_dir,
        chunk_days=config.day_chunk_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve NDVI")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file.')
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(args.config)
    main(cfg)
