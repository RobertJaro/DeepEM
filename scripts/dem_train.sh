#!/bin/bash
#SBATCH --job-name=dem                # Job name
#SBATCH --partition=gpu                      # Queue name
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks=16                           # Run 4 tasks
#SBATCH --mem=64000                          # Job memory request in Megabytes
#SBATCH --gpus=2                             # Number of GPUs
#SBATCH --time=24:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec
#SBATCH --output=/gpfs/gpfs0/robert.jarolim/dem/logs/%j.log     # Standard output and error log

module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/DEM
python3 -i -m dem.train.train --n_dims 512 --base_dir /gpfs/gpfs0/robert.jarolim/dem/version18_error --lambda_l1 1e-2 --temperature_response /gpfs/gpfs0/robert.jarolim/data/dem/aia_temperature_response_2013.csv --data_path /gpfs/gpfs0/robert.jarolim/data/dem --converted_path /gpfs/gpfs0/robert.jarolim/data/converted/dem_prep
