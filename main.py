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

L = 20 * (1 + n // 5) # 20, 40, 60, 80, 100
p_error = 2 ** (6 + n % 5) # 64, 128, 256, 512, 1024
η = 0.1
c = 16
T = L
shots = 10000

matching = Matching(pcm(L))

def f(*_):
    mystate = init_state(L)

    density = decoder_2D_density(mystate, T, c, η, 1 / p_error)

    correction = mwpm(matching, mystate.q)
    ca_mwpm_fail = logical_error(correction ^ mystate.error)
    return ca_mwpm_fail, density

with Pool(num_cpus) as p:
    result = p.map(f, range(shots))

result_array = np.array(result, dtype = object).mean(axis = 0)

np.save(f"data/run_4/run_4_{L}_{p_error}.npy", result_array)

elapsed = time() - t0
print(f"Job for L={L} and p=1/{p_error} took time:")
print(strftime("%H:%M:%S", gmtime(elapsed)))
