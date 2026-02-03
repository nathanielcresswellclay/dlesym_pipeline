import xarray as xr
import omegaconf
from dask.diagnostics import ProgressBar
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(infile, outfile, time_chunks):

    ds = xr.open_zarr(infile)
    logger.info(f"Rechunking along time with chunk size {time_chunks}...")
    ds = ds.chunk({"time": time_chunks})

    logger.info(f"Saving rechunked dataset to {outfile}...")
    with ProgressBar():
        ds.to_zarr(outfile, mode="w", consolidated=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rechunk a zarr dataset along the time dimension."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = omegaconf.OmegaConf.load(args.config)
    logger.info(f"Loaded configuration {config}")
    main(**config)
