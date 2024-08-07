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
    description = 'Run the ???.',
    epilog = 'Saves ??? to the current directory.'
)
parser.add_argument('-n', type = int, default = 0)
args = parser.parse_args()
n = args.n

L = 20 * (1 + n // 5) # 20, 40, 60, 80, 100, 120, 140, 160, 180, 200
p_error = 0.01 * (1 + n % 5) # 0.01, 0.02, 0.03, 0.04, 0.05
eta = 0.1
c = 16
T = L
shots = 10000

matching = Matching(pcm(L))

def f(n):
    mystate = init_state(L)

    decoder_2D(mystate, T, c, eta, p_error = p_error, history = False)

    correction = mwpm(matching, mystate.q)
    ca_mwpm_fail = logical_error(correction ^ mystate.error)
    return ca_mwpm_fail, mystate.N / L ** 2

with Pool(num_cpus) as p:
    result = p.map(f, range(shots))

fail_array = np.mean(result, axis = 0)
np.save(f"data/run_3_{L}_{p_error}.npy", fail_array)
