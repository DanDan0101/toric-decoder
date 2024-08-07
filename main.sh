#!/usr/bin/bash
#SBATCH --job-name=main
#SBATCH --time=1:00:00
#SBATCH -p owners
#SBATCH --array=0-24
#SBATCH -c 8
#SBATCH --mail-type=ALL

python3 main.py -n $SLURM_ARRAY_TASK_ID
