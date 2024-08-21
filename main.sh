#!/usr/bin/bash
#SBATCH --job-name=main
#SBATCH --time=1:30:00
#SBATCH -p owners
#SBATCH --array=0-54
#SBATCH -G 1
#SBATCH --mail-type=ALL

ml load py-cupy/12.1.0_py39
ml load py-nvidia-ml-py/12.550.52_py39

python3 main.py -n $SLURM_ARRAY_TASK_ID
