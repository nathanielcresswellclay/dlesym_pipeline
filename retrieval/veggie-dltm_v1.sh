#!/bin/bash
#SBATCH --job-name=era5_retrieval
#SBATCH --account=m4935
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # Perlmutter CPU nodes have 32 cores per socket
#SBATCH --time=00:10:00
#SBATCH --constraint=cpu             # Ensures we use a CPU node, not GPU
#SBATCH --qos=debug
#SBATCH --output=/pscratch/sd/n/nacc/veggie-dltm/dlesym_pipeline/retrieval/output/era5_retrieval.%j.out
#SBATCH --error=/pscratch/sd/n/nacc/veggie-dltm/dlesym_pipeline/retrieval/output/era5_retrieval.%j.err

# activate environment
source /pscratch/sd/n/nacc/miniconda/etc/profile.d/conda.sh
conda activate sh_env_wyik

echo "Starting ERA5 retrieval script..."
# Run the Python script
# python era5_arco_retrieval.py --config veggie-dltm-v1.yaml