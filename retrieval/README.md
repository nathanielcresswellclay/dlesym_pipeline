# Running in Interactive mode
Running in interactive mode is mostly useful for debugging utilities. First get your CPU allocation:

`salloc --account=m4935 --nodes 1 --ntasks-per-node 4 --constraint cpu --qos interactive --time 01:00:00`

Then activate your conda environment, sourcing first if it is not initialized in you shell program: 

```
source /pscratch/sd/n/nacc/miniconda/etc/profile.d/conda.sh
conda activate sh_env_wyik
```

Make sure your working directory is dlesym_pipline. And finally run you python utility:  

`python ./retrieval/utils/era5_arco_retrieval.py --config ./retrieval/configs/arco_retrieval_dev.yaml`


# ERA5 retrieval sources

Two utilities share the same request schema (`variable_name`, optional `level`,
`time_start`/`time_end`/`time_freq`, `accumulation_period`, `constant`,
`output_file`, `overwrite`) and produce interchangeable netcdf output:

- `era5_arco_retrieval.py` — Google ARCO-ERA5 (anonymous, fast, no account).
- `era5_cds_retrieval.py` — Copernicus CDS via `cdsapi`. Requires a CDS account,
  accepted dataset licences, and a `~/.cdsapirc` file. Requests are issued
  per month and queued by CDS, so large jobs can take a while.

`python ./retrieval/utils/era5_cds_retrieval.py --config ./retrieval/configs/cds_retrieval_dev.yaml`

Accumulated variables (`accumulation_period`) are fetched hourly (with a
look-back padding) and summed over the trailing window, matching the arco utility.
