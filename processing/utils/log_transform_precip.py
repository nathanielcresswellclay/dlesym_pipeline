import xarray as xr
import numpy as np
import omegaconf
from dask.diagnostics import ProgressBar
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(infile, outfile, eps=1e-8):

    ds = xr.open_zarr(infile, chunks='auto')

    precip = ds.inputs.sel(channel_in='tp6-48H')
    # stable log transform
    precip_log = np.log(precip + eps) - np.log(eps)

    # add as new channel (keep original tp6-48H unchanged)
    new_channel_name = f"tp6-48H_lt{eps}"
    precip_log_da = precip_log.expand_dims("channel_in").assign_coords(
        channel_in=[new_channel_name]
    )
    ds = ds.assign(
        inputs=xr.concat([ds.inputs, precip_log_da], dim="channel_in")
    )

    logger.info(f"Saving dataset with new channel '{new_channel_name}' to {outfile}...")
    with ProgressBar():
        ds.to_zarr(outfile, mode='w', consolidated=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interpolate using nearest neighbor.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    
    # open config
    config = omegaconf.OmegaConf.load(args.config)
    logger.info(f"Loaded configuration {config}")
    # Load the configuration
    main(**config)

