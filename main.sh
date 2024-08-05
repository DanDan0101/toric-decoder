#!/usr/bin/bash
#SBATCH --job-name=main
#SBATCH --time=4:00:00
#SBATCH -p hns
#SBATCH --array=0-59
#SBATCH -c 8
#SBATCH --mail-type=ALL

python3 main.py -n $SLURM_ARRAY_TASK_ID
