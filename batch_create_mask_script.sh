#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A Berzelius-2025-212
#SBATCH --gpus=1
#SBATCH -t 00-02:00:00
#SBATCH -C thin

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate thesis

# Execute your code
python pipnet/scripts/data_create_mask.py