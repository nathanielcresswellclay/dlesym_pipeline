import os
import time
import logging
import numpy as np
import pandas as pd
import xarray as xr
import omegaconf
from dask.diagnostics import ProgressBar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_time_subset(cfg):
    time_subset = (
        pd.date_range(
            start=cfg.time_subset_start,
            end=cfg.time_subset_end,
            freq=cfg.time_subset_freq,
        )
        .to_numpy()
        .astype("datetime64[ns]")
    )
    logger.info(f"Selecting explicit time subset ({len(time_subset)} entries)")
    return time_subset


def _write_variable(
    *,
    files,
    var_name,
    channel_dim,
    channel_selector,
    output_file,
    time_subset=None,
    chunks=None,
):
    """
    Stream one channel block at a time into a Zarr store.
    """
    first = True

    for f in files:
        logger.info(
            f"Processing {var_name} from {f.path} "
            f"(channels={getattr(f, channel_dim)})"
        )

        ds = xr.open_zarr(f.path, chunks=chunks)[var_name]

        # explicit channel selection
        ds = ds.sel({channel_dim: getattr(f, channel_dim)})

        # explicit time indexing (kept intentionally)
        if time_subset is not None and "time" in ds.dims:
            ds = ds.sel(time=time_subset)

        # ensure string channels
        ds[channel_dim] = ds[channel_dim].astype(object)

        # enforce chunking
        if chunks is not None:
            ds = ds.chunk(chunks)

        # clear inherited encoding so Zarr uses Dask chunks
        ds.encoding.clear()

        mode = "w" if first else "a"
        append_dim = None if first else channel_dim

        logger.info(
            f"Writing {var_name} "
            f"(mode={mode}, append_dim={append_dim})"
        )

        with ProgressBar():
            ds.to_zarr(
                output_file,
                mode=mode,
                append_dim=append_dim,
                consolidated=False,
            )

        first = False


def main(config: str):
    """
    Compile separate trainsets into a single Zarr file
    with bounded memory usage.
    """
    cfg = omegaconf.OmegaConf.load(config)

    if os.path.exists(cfg.output_file) and not cfg.overwrite:
        logger.info(
            f"Output file {cfg.output_file} already exists. "
            f"Set overwrite=True to overwrite."
        )
        return

    t0 = time.perf_counter()

    # build explicit time index ONCE
    time_subset = _build_time_subset(cfg)

    # ---- inputs ----
    _write_variable(
        files=cfg.input_files,
        var_name="inputs",
        channel_dim="channel_in",
        channel_selector="channel_in",
        output_file=cfg.output_file,
        time_subset=time_subset,
        chunks=dict(cfg.chunks),
    )

    # ---- targets ----
    _write_variable(
        files=cfg.output_files,
        var_name="targets",
        channel_dim="channel_out",
        channel_selector="channel_out",
        output_file=cfg.output_file,
        time_subset=time_subset,
        chunks=dict(cfg.chunks),
    )

    # ---- constants ----
    _write_variable(
        files=cfg.constant_files,
        var_name="constants",
        channel_dim="channel_c",
        channel_selector="channel_c",
        output_file=cfg.output_file,
        time_subset=None,  # constants are time-invariant
        chunks={k: -1 for k in dict(cfg.constant_chunks)}
        if hasattr(cfg, "constant_chunks")
        else None,
    )

    logger.info(
        f"Finished writing merged dataset in "
        f"{time.perf_counter() - t0:.2f} seconds"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile separate trainsets into a single Zarr file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    main(config=args.config)
