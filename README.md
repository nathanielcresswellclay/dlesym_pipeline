# dlesym_pipeline

Pipeline for preparing training and initialization data for DLESyM-style
deep-learning climate models. Most workflows follow the same three-stage
pattern: **retrieve** raw inputs from an upstream archive, **process** them
into model-ready fields (derived variables, regridding, imputation, etc.),
and **compile** the result into a single Zarr training/init dataset.

## Repo structure

```
dlesym_pipeline/
├── retrieval/        # download raw data (ERA5, ERA5-Land, MAIAC, SMAP, ...)
│   ├── utils/        #   driver scripts (one per data source)
│   └── configs/      #   YAML request lists
├── processing/       # derive variables, regrid, impute, log/scale transforms
│   ├── utils/
│   └── configs/
├── compilation/      # bundle processed fields into a single training Zarr
│   ├── utils/
│   └── configs/
├── aimip/            # AIMIP init-data pipeline (see aimip/README.md)
├── dlesym_pipeline.yml   # conda environment spec
├── veggie-dltm_v4.sh     # end-to-end driver for the DLTM v4 training set
└── README.md
```

Every stage script takes a single `--config <path.yaml>` argument. Driver
shell scripts (e.g. `veggie-dltm_v4.sh`, `aimip/aimip_init-prep.sh`) chain the
stages together for a specific dataset.

> ⚠️ All absolute paths inside the configs and driver scripts contain
> `nacc` (e.g. `/home/disk/brume/nacc/...`) and point at the author's
> filesystem. Edit them to directories you have read/write access to before
> running anything.

## Environment

The pipeline runs in a conda environment called `dlesym_pipeline`, defined
by [`dlesym_pipeline.yml`](dlesym_pipeline.yml) (Python 3.10, PyTorch +
CUDA 11.7, xarray, zarr, healpy, cdsapi, hydra/omegaconf, dask, and the
usual scientific Python stack).

To create it:

```bash
conda env create -f dlesym_pipeline.yml
conda activate dlesym_pipeline
```

Then, before running any stage script, point `PYTHONPATH` at the repo root
so the cross-stage imports (e.g. `from compilation.utils.write_zarr import
plot_healpix`) resolve:

```bash
export PYTHONPATH=/path/to/dlesym_pipeline
```

The driver shell scripts do this automatically.

## Where to start

- For the AIMIP collaborative initialization workflow, see
  [`aimip/README.md`](aimip/README.md).
- For an end-to-end DLTM-v4 training-data build, see
  [`veggie-dltm_v4.sh`](veggie-dltm_v4.sh).
