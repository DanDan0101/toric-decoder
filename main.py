import sys
sys.path.insert(0, 'toric-decoder')

import os
from multiprocess import Pool
num_cpus = len(os.sched_getaffinity(0))

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(
    description = 'Run the ???.',
    epilog = 'Saves ??? to the current directory.'
)
parser.add_argument('-t', type = int, default = 1)
args = parser.parse_args()
t = args.t

with Pool(num_cpus) as pool:
    pass