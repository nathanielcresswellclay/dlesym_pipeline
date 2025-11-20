#!/bin/bash

# activate environment
source /home/disk/brume/nacc/anaconda3/etc/profile.d/conda.sh
conda activate dlesym-1.0

# retrieve era5 data
python retrieval/utils/era5_arco_retrieval.py --config retrieval/configs/veggie-dltm_era5.yaml

# retrieve era5-land data
python retrieval/utils/era5-land_retrieval.py --config retrieval/configs/veggie-dltm_era5_land.yaml
# impute ERA5-Land over oceans 

# retrieve MAIAC data
python retrieval/utils/retrieve_maiac.py --config retrieval/configs/veggie-dltm_maiac_retrieval.yaml