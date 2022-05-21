#!/bin/bash
#SBATCH --job-name=dem                # Job name
#SBATCH --partition=gpu_devel                      # Queue name
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks=4                           # Run 4 tasks
#SBATCH --mem=24000                          # Job memory request in Megabytes
#SBATCH --gpus=1                             # Number of GPUs
#SBATCH --time=12:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec
#SBATCH --output=/gpfs/gpfs0/robert.jarolim/dem/logs/%j.log     # Standard output and error log

module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/DEM
python3 -i -m dem.data.download_flare_events --download_dir /gpfs/gpfs0/robert.jarolim/data/dem --flare_list /gpfs/gpfs0/robert.jarolim/data/dem/m_flare_list.csv --email robert.jarolim@uni-graz.at
