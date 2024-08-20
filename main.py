from time import time
# from time import strftime, gmtime
t0 = time()

import sys
sys.path.insert(0, 'toric-decoder')

import numpy as np
from toric import State, decoder_2D, mwpm, pcm, logical_error
from pymatching import Matching

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(
    description = 'Run the simulation.',
    epilog = 'Saves data to the ./data/ directory.'
)
parser.add_argument('-n', type = int, default = 0)
args = parser.parse_args()
n = args.n # TODO
debug = (n == -1)

L = 100

N = int(200000000/L**2) # Assuming 16 GB of GPU memory
if debug:
    N = int(75000000/L**2) # Assuming 6 GB of GPU memory

R = int(50*(L/100)**2) # Repetitions
if R < 1 or debug:
    R = 1

p_error = 0.005
η = 0.1
c = 16
T = L

matching = Matching(pcm(L))

state = State(N, L)

decoder_2D(state, T, c, η, p_error)

fails = np.empty(R)

for i in range(R):
    x_correction, y_correction = mwpm(matching, state.q)
    fails[i] = logical_error(x_correction ^ state.x_error.get(), y_correction ^ state.y_error.get()).mean()

fail_rate = fails.mean() # Just a single float

print(f"Fail rate: {fail_rate}")

# np.save(f"data/run_9/run_9_{L}_{p_error}_{n}.npy", fail_rate)

elapsed = time() - t0
# print(f"Job for L={L} and p={p_error / 10000} took time:")
# print(strftime("%H:%M:%S", gmtime(elapsed)))
print(int(elapsed))
