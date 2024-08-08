from time import time, strftime, gmtime
t0 = time()

import sys
sys.path.insert(0, 'toric-decoder')

import numpy as np
from toric import *
from pymatching import Matching

import os
from multiprocess import Pool
num_cpus = len(os.sched_getaffinity(0))

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(
    description = 'Run the simulation.',
    epilog = 'Saves data to the ./data/ directory.'
)
parser.add_argument('-n', type = int, default = 0)
args = parser.parse_args()
n = args.n

L = 100 * (1 + n // 10) # 100, 200, 300, ..., 500
p_error = (1 + n % 10) # 1, 2, 3, ..., 10
η = 0.1
c = 16
T = L
shots = 10000

# matching = Matching(pcm(L))

def f(*_):
    mystate = init_state(L)

    decoder_2D(mystate, T, c, η, 0.5 ** p_error)

    # correction = mwpm(matching, mystate.q)
    # ca_mwpm_fail = logical_error(correction ^ mystate.error)
    return mystate.N / L**2

with Pool(num_cpus) as p:
    result = p.map(f, range(shots))

density = np.mean(result)

np.save(f"data/run_5/run_5_{L}_{p_error}.npy", density)

elapsed = time() - t0
print(f"Job for L={L} and p=1/{int(2**p_error)} took time:")
print(strftime("%H:%M:%S", gmtime(elapsed)))
