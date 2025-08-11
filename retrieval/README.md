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
