#!/bin/bash

# activate environment
source /home/disk/brume/nacc/anaconda3/etc/profile.d/conda.sh
conda activate dlesym-1.0

###############################################################################
#####################    RETRIEVAL OF INPUT DATA    ###########################
###############################################################################

# retrieve era5 data
python retrieval/utils/era5_arco_retrieval.py --config retrieval/configs/veggie-dltm_era5.yaml

# retrieve era5-land data
python retrieval/utils/era5-land_retrieval.py --config retrieval/configs/veggie-dltm_era5_land.yaml

# retrieve MAIAC data
python retrieval/utils/retrieve_maiac.py --config retrieval/configs/veggie-dltm_ndvi_retrieval.yaml

###############################################################################
#####################    DATA PROCESSING PIPELINE    ##########################
###############################################################################

# processing ERA5-land data by imputing and remapping 
python processing/utils/constant_impute_ocean.py --config processing/configs/veggie-dltm_constant_impute.yaml

# calculat q2m from specific humidity and dewpoint temperature
python processing/utils/calculate_q2m.py --config processing/configs/veggie-dltm_calculate_q2m.yaml