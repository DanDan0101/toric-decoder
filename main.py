from time import time
# from time import strftime, gmtime
t0 = time()

import sys
sys.path.insert(0, 'toric-decoder')

# import numpy as np
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

N = 20000
L = 100
p_error = 0.004
η = 0.1
c = 16
T = L

matching = Matching(pcm(L))

state = State(N, L)

decoder_2D(state, T, c, η, p_error)

x_correction, y_correction = mwpm(matching, state.q)
fail = logical_error(x_correction ^ state.x_error.get(), y_correction ^ state.y_error.get())

fail_rate = fail.mean() # Just a single float

print(f"Fail rate: {fail_rate}")

# np.save(f"data/run_9/run_9_{L}_{p_error}_{n}.npy", fail_rate)

elapsed = time() - t0
# print(f"Job for L={L} and p={p_error / 10000} took time:")
# print(strftime("%H:%M:%S", gmtime(elapsed)))
print(int(elapsed))
