#!/usr/bin/bash
#SBATCH --job-name=main
#SBATCH --time=3:00:00
#SBATCH -p owners
#SBATCH --array=0-9
#SBATCH -G 1
#SBATCH --mail-type=ALL

ml load py-cupy/12.1.0_py39

python3 main.py -n $SLURM_ARRAY_TASK_ID
