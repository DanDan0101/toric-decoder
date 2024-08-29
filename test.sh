#!/usr/bin/bash
#SBATCH --job-name=test
#SBATCH --time=1:30:00
#SBATCH -p owners
#SBATCH --array=0-44:11
#SBATCH -G 1
#SBATCH --mail-type=ALL

ml load py-cupy/12.1.0_py39
ml load py-nvidia-ml-py/12.550.52_py39

python3 -u main.py -n $SLURM_ARRAY_TASK_ID
