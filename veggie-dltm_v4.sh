#!/bin/bash

# activate environment
source /home/disk/brume/nacc/anaconda3/etc/profile.d/conda.sh
conda activate dlesym_pipeline
export PYTHONPATH=/home/disk/brume/nacc/veggie-dltm/dlesym_pipeline 

###############################################################################
#####################    RETRIEVAL OF INPUT DATA    ###########################
###############################################################################

# retrieve era5 data
python retrieval/utils/era5_arco_retrieval.py --config retrieval/configs/veggie-dltm_era5.yaml

# retrieve era5-land data
python retrieval/utils/era5-land_retrieval.py --config retrieval/configs/veggie-dltm_era5_land.yaml

# retrieve MAIAC data keeping this commented since I couldn't find an elegent way to 
# skip previously downloaded files. Instead earthaccess produces sevreal lines of output
# for each existing file and pollutes logs. 
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

# map to hpx64 ERA5 data for coupling, ERA5-land and MAIAC NDVI for DLTM state variables, and constant inputs
python processing/utils/map2hpx.py --config /home/disk/brume/nacc/veggie-dltm/dlesym_pipeline/processing/configs/veggie-dltm_remap.yaml

# Trailing average calculation for DLTM forcing variables. Here we do 48H trailing average
python processing/utils/trailing_average.py --config processing/configs/veggie-dltm_trailing_average_atmos.yaml

# mask out ice sheets 
 python processing/utils/mask_icesheets.py --config processing/configs/veggie-dltm_mask_icesheets.yaml

# calculat statistics on ndvi, sm and st for approximating vegetation/land type
python processing/utils/compute_var_meta.py --config processing/configs/veggie-dltm_land-constants.yaml

###############################################################################
###########################     Compilation     ###############################
###############################################################################

# compile all variables into one training dataset
python compilation/utils/write_zarr.py --config compilation/configs/veggie-dltm_zarr-compile.yaml




