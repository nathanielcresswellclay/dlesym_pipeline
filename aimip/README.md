# AIMIP Initialization-Data Pipeline

This directory contains the configs and helper scripts used to prepare
**initial-condition data** for the AIMIP collaborative modelling effort. The
pipeline pulls a short window of ERA5 reanalysis from the ARCO archive,
derives a few diagnostic variables, remaps everything to the HEALPix-64 grid, 
and writes a single Zarr store that can be fed directly to the model as an 
initialization dataset.

The full pipeline is driven by [`aimip_init-prep.sh`](aimip_init-prep.sh),
which simply chains the same generic stages used elsewhere in
`dlesym_pipeline/` (retrieval → processing → compilation), but with
AIMIP-specific YAML configs that all live in this folder.

> ⚠️ **User-specific file paths.** Every absolute path in this directory
> that contains `nacc` (for example `/home/disk/brume/nacc/...` or
> `/home/disk/mercury2/nacc/AIMIP2026/...`) is specific to the author's
> environment and **will not exist on your machine**. Before running
> anything, do a pass through:
>
> - `aimip_init-prep.sh` — update the `source .../conda.sh`,
>   `conda activate`, `cd`, and `PYTHONPATH` lines, and the
>   `--ref-dataset` / `--output-dataset` paths passed to
>   `create_dummy_ocean_init.py`.
> - Every `aimip_1978-init*.yaml` in this folder — update input/output
>   NetCDF paths, Zarr output paths, plot directories, and any reference
>   datasets (notably the ISCCP OLR file and the SST / LSM / topography
>   constants in `aimip_1978-init_zarr-write.yaml`).
>
> Replace each `nacc`-bearing path with a directory you have read/write
> access to. The directory layout otherwise does not matter — the pipeline
> only requires that the outputs of one stage point at the inputs of the
> next.

## Quick start

```bash
# from anywhere on the system
bash ./aimip/aimip_init-prep.sh
```

The wrapper script handles environment activation
(`conda activate dlesym_pipeline`), sets `PYTHONPATH`, and runs every stage in
order. Each stage is idempotent: configs default to `overwrite: False`, so a
re-run will skip steps whose outputs already exist.

## Conventions used by the AIMIP configs

All AIMIP configs in this folder share a common set of conventions, so once
you understand one you can read the rest at a glance:

- **Time window.** Configs target a 1-week window starting
  `1978-10-01T00:00:00` at 3-hourly frequency. To prepare a different
  initialization, search-and-replace the `time_start` / `time_end` /
  `time.start` / `time.end` / `times.start` / `times.end` blocks consistently
  across every config.
- **Naming.** Files follow the prefix
  `aimip_<YYYY>-init_<step>.yaml`, e.g. `aimip_1978-init_windspeed.yaml`.
- **Output root.** All intermediate and final outputs land under
  `/home/disk/mercury2/nacc/AIMIP2026/init_data/`. 
- **Grid.** Everything is remapped to HEALPix `nside=64` (`HPX64`) with
  bilinear interpolation.
- **Sanity-check plots.** Most steps write a small set of PNGs to a
  `*_frames` subdirectory so each transform can be eyeballed before
  compilation.

## Important notes

```
  ┌──────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
  │ ERA5 (ARCO)  │ →  │ derive + remap (HPX) │ →  │ compile training    │
  │  retrieval   │    │  windspeed, tau,     │    │ Zarr + dummy ocean  │
  │              │    │  ll2hpx, ttr→olr     │    │ init                │
  └──────────────┘    └──────────────────────┘    └─────────────────────┘
```

`dlesym_pipeline/aimip/aimip_init-prep.sh` contains the entire pipeline for the AIMIP init should be used as a basis for understanding proper preparation of initialzation data for DLESyM. Below are a couple of notes about potentially non-intuitive parts of that pipeline: 

### Rescale TTR to look like ISCCP OLR

In `dlesym_pipeline/aimip/aimip_init-prep.sh` this is where we turn TTR into OLR: 

```bash
python processing/utils/transform_ttr1h_olr.py \
    --config aimip/aimip_1978-init_transform_ttr1h-olr.yaml
```

DLESyM was trained on ISCCP OLR rather than on ERA5's TTR1H. This step
matches the moments of HPX64 `ttr1h` to those of an ISCCP OLR reference
dataset over comparable days of year, producing `ttr1h_HPX64_olr.nc`.
Note that the reference OLR climatology covers 1983–2017, so the script
internally pairs the 1978 target days with same-day-of-year statistics from
the available reference window (`plotting.source_times` /
`plotting.target_times` make this explicit). A
`constraints.min_quantile` / `global_floor` block clips physically
unrealistic low values.

### Compile the training-format Zarr

```bash
python compilation/utils/write_zarr.py \
    --config aimip/aimip_1978-init_zarr-write.yaml
```

Bundles every HPX64 NetCDF — plus the long-term SST forcing file and the LSM
/ topography constants — into a single Zarr store
`aimip_1978-init.zarr` under `dst_directory`. The `inputs:` and `outputs:`
mappings tell `write_zarr.py` which variables become model inputs vs.
outputs (here they are identical), and the `time_dim` / `time_slice`
blocks pin the dataset to the chosen 1978 window at 3 h frequency.

### Build a dummy ocean-init Zarr

```bash
python aimip/create_dummy_ocean_init.py \
    --ref-dataset  /home/disk/mercury2/nacc/AIMIP2026/init_data/aimip_1978-init.zarr \
    --output-dataset /home/disk/mercury2/nacc/AIMIP2026/init_data/aimip_1978-init_ocean.zarr
```

DLESyM's coupled ocean module expects an ocean-init dataset with specific
coupling channels (`z1000-48H`, `ws10-48H`, `olr-48H`). For AIMIP we are not
actually initializing the ocean module from observed state, but the channels
still need to exist so the model can be instantiated. `create_dummy_ocean_init.py`
clones the atmospheric init Zarr, renames the relevant input channels in
place, and writes the result as `aimip_1978-init_ocean.zarr`. The
underlying values are placeholders and should not be used as physical
fields.

## Final outputs

After a successful run, the artefacts you actually hand off are:

- `aimip_1978-init.zarr` — atmospheric initialization Zarr (model inputs +
  outputs + constants on HPX64, 3 h cadence over the chosen week).
- `aimip_1978-init_ocean.zarr` — dummy ocean-init Zarr with the correct
  coupling channels.

Everything else under `/home/disk/mercury2/nacc/AIMIP2026/init_data/` is an
intermediate NetCDF or a QC plot directory and can be regenerated from this
pipeline.

## Files in this directory

| File                                             | Purpose                                                              |
| ------------------------------------------------ | -------------------------------------------------------------------- |
| `aimip_init-prep.sh`                             | End-to-end driver script                                             |
| `aimip_1978-init.yaml`                           | ERA5/ARCO retrieval requests                                         |
| `aimip_1978-init_windspeed.yaml`                 | 10 m windspeed derivation                                            |
| `aimip_1978-init_calculate_tau300-700.yaml`      | 300–700 hPa thickness derivation                                     |
| `aimip_1978-init_ll2hpx.yaml`                    | Lat-lon → HEALPix-64 remap list                                      |
| `aimip_1978-init_transform_ttr1h-olr.yaml`       | TTR → ISCCP-OLR moment matching                                      |
| `aimip_1978-init_zarr-write.yaml`                | Final training-format Zarr compilation                               |
| `create_dummy_ocean_init.py`                     | Ocean-init Zarr generator (channels only)                            |
