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
n = args.n # 0 - 54
debug = (n == -1)

L = int(100 * (n // 11 + 1)) # L = 100, 200, 300, 400, 500
p_error = ((n % 11) + 35) / 10000 # p_error = 0.0035, 0.0036, ..., 0.0045

N = int(24000/(L/100)**2) # Assuming 16 GB of GPU memory
if debug:
    N = int(9000/(L/100)**2) # Assuming 6 GB of GPU memory

# R = int(420*(L/100)**2) # Repetitions, statistical
R = int(40 / (L/100)) # Ensure total runtime of about an hour
if R < 1 or debug:
    R = 1
    print("Warning: R < 1, setting R = 1")

η = 0.1
c = 16
T = L

matching = Matching(pcm(L))

fails = np.empty(R)

for i in range(R):
    state = State(N, L)
    decoder_2D(state, T, c, η, p_error)
    x_correction, y_correction = mwpm(matching, state.q)
    fails[i] = logical_error(x_correction ^ state.x_error.get(), y_correction ^ state.y_error.get()).mean()

fail_rate = fails.mean() # Just a single float

np.save(f"data/run_10/run_10_{n}.npy", fail_rate)

elapsed = time() - t0
# print(f"Job for L={L} and p={p_error / 10000} took time:")
# print(strftime("%H:%M:%S", gmtime(elapsed)))
print(int(elapsed))
