#!/bin/bash

# activate environment
source /home/disk/brume/nacc/anaconda3/etc/profile.d/conda.sh
conda activate dlesym_pipeline
cd /home/disk/brume/nacc/veggie-dltm/dlesym_pipeline
export PYTHONPATH=/home/disk/brume/nacc/veggie-dltm/dlesym_pipeline 

###############################################################################
#####################    RETRIEVAL OF INPUT DATA    ###########################
###############################################################################

# retrieve era5 data
python retrieval/utils/era5_arco_retrieval.py --config aimip/aimip_1978-init.yaml

###############################################################################
###########################    Processing...    ###############################
###############################################################################

# calculate windspeed from u and v components
python processing/utils/calculate_windspeed.py --config aimip/aimip_1978-init_windspeed.yaml

# calculate tau300-700 from z300 and z700 components
python processing/utils/calculate_tau300-700.py --config aimip/aimip_1978-init_calculate_tau300-700.yaml

# map to hpx64
python processing/utils/map2hpx.py --config aimip/aimip_1978-init_ll2hpx.yaml

# transform ttr1h to olr
python processing/utils/transform_ttr1h_olr.py --config aimip/aimip_1978-init_transform_ttr1h-olr.yaml

###############################################################################
##################     Compilation & Zarr Manipulation   ######################
###############################################################################

# compile all variables into one training dataset
python compilation/utils/write_zarr.py --config aimip/aimip_1978-init_zarr-write.yaml

# create dummy ocean init dataset (this is necessary to initialize forcing ocean module but won't contain any data)
python aimip/create_dummy_ocean_init.py --ref-dataset /home/disk/mercury2/nacc/AIMIP2026/init_data/aimip_1978-init.zarr --output-dataset /home/disk/mercury2/nacc/AIMIP2026/init_data/aimip_1978-init_ocean.zarr
