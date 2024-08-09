#!/usr/bin/bash
#SBATCH --job-name=main
#SBATCH --time=2:00:00
#SBATCH -p owners
#SBATCH --array=0-49
#SBATCH -c 8
#SBATCH --mail-type=ALL

python3 main.py -n $SLURM_ARRAY_TASK_ID
