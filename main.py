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

L = 20 * (1 + n // 10) # 20, 40, 60, ..., 100
p_error = 4 # 0.004
η = 0.1
c = 16
T = L
shots = 100000

matching = Matching(pcm(L))

def f(*_):
    mystate = init_state(L)

    decoder_2D(mystate, T, c, η, p_error / 1000)

    correction = mwpm(matching, mystate.q)
    ca_mwpm_fail = logical_error(correction ^ mystate.error)
    return ca_mwpm_fail

with Pool(num_cpus) as p:
    result = p.map(f, range(shots))

fail_rate = np.array(result).mean(axis = 0) # Just a single float

np.save(f"data/run_7/run_7_{L}_{n % 10}.npy", fail_rate)

elapsed = time() - t0
print(f"Job for L={L} and p={p_error / 1000} took time:")
print(strftime("%H:%M:%S", gmtime(elapsed)))
