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

L = 20 * (1 + n // 5)
p_error = 0.05
eta = 0.1
c = 2 ** (1 + n % 5)
T = L
shots = 10000

matching = Matching(pcm(L))

def f(n):
    mystate = init_state(L, p_error)
    initial_correction = mwpm(matching, mystate.q)
    mwpm_fail = logical_error(initial_correction ^ mystate.error)

    decoder_2D(mystate, T, c, eta, p_error = 0, history = False)
    ca_fail = (mystate.N > 0 or logical_error(mystate.error))

    correction = mwpm(matching, mystate.q)
    ca_mwpm_fail = logical_error(correction ^ mystate.error)
    return mwpm_fail, ca_fail, ca_mwpm_fail

with Pool(num_cpus) as p:
    result = p.map(f, range(shots))

fail_array = np.sum(result, axis = 0) / shots # mwpm_fail, ca_fail, ca_mwpm_fail
np.save(f"data/fail_array_{L}_{c}.npy", fail_array)
