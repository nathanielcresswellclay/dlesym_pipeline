#!/bin/bash

# activate environment
source /home/disk/brume/nacc/anaconda3/etc/profile.d/conda.sh
conda activate dlesym-1.0-metpy

###############################################################################
#####################    RETRIEVAL OF INPUT DATA    ###########################
###############################################################################

# retrieve era5 data
python retrieval/utils/era5_arco_retrieval.py --config retrieval/configs/veggie-dltm_era5.yaml

# retrieve era5-land data
python retrieval/utils/era5-land_retrieval.py --config retrieval/configs/veggie-dltm_era5_land.yaml

# retrieve MAIAC data
# python retrieval/utils/retrieve_maiac.py --config retrieval/configs/veggie-dltm_ndvi_retrieval.yaml

###############################################################################
###########################    Processing...    ###############################
###############################################################################

# processing ERA5-land data by imputing
python processing/utils/constant_impute_ocean.py --config processing/configs/veggie-dltm_constant_impute.yaml

# calculate q2m from specific humidity and dewpoint temperature
python processing/utils/calculate_q2m.py --config processing/configs/veggie-dltm_calculate_q2m.yaml

# consolidate NDVI-MAIAC into standard format
python processing/utils/processing_maiac.py --config processing/configs/veggie-dltm_processing_maiac.yaml
# impute maiac temporally and with constants over ocean 
python processing/utils/impute_maiac.py --config processing/configs/veggie-dltm_maiac_impute.yaml

# calculate spatially resolved variable metadata (min, max, annual range) to be used for constant inputs 
python /home/disk/brume/nacc/veggie-dltm/dlesym_pipeline/processing/utils/compute_var_meta.py --config /home/disk/brume/nacc/veggie-dltm/dlesym_pipeline/processing/configs/veggie-dltm_land-constants.yaml

# map to hpx64 ERA5 data for coupling, ERA5-land and MAIAC NDVI for DLTM state variables, and constant inputs
python /home/disk/brume/nacc/veggie-dltm/dlesym_pipeline/processing/utils/map2hpx.py --config /home/disk/brume/nacc/veggie-dltm/dlesym_pipeline/processing/configs/veggie-dltm_remap.yaml


# generate mask for prognostic domain 

# apply mask to all land-defined fields. Will constrain spurious points over land: greenland and antarctica

###############################################################################
###########################     Compilation     ###############################
###############################################################################

# compile all variables into one training dataset 




