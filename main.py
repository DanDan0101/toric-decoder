from time import time
from time import strftime, gmtime
t0 = time()
TIMELIMIT = 3600 # 1 hour

import sys
sys.path.insert(0, 'toric-decoder')

import numpy as np
import cupy as cp
from toric import State, decoder_2D, mwpm, pcm, logical_error
from pymatching import Matching

import os
from pynvml import nvmlInit, nvmlDeviceGetHandleByUUID, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo
nvmlInit()
try:
    handle = nvmlDeviceGetHandleByUUID(os.getenv("CUDA_VISIBLE_DEVICES"))
except:
    handle = nvmlDeviceGetHandleByIndex(0)
name = nvmlDeviceGetName(handle)
mem = nvmlDeviceGetMemoryInfo(handle).free / 1000**3 # GB of VRAM
print(f"Running on {name} with {mem:.2f} GB of VRAM.")

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

if debug:
    L = 50
    p_error = 0.004
else:
    L = int(100 * (n // 11 + 1)) # L = 100, 200, 300, 400, 500
    p_error = ((n % 11) + 35) / 10000 # p_error = 0.0035, 0.0036, ..., 0.0045

p_error = cp.float32(p_error)

N = int(3125/(L/100)**2 * mem) # 32 N L^2 < mem ??? WHY 32?
R = int(10**7/N) # Repetitions, statistical
# R = int(40 * (TIMELIMIT/3600) / (L/100)) # Assuming ~90s * L/100 per repetition
if R < 1 or debug:
    R = 1
    print("Warning: setting R = 1")

η = 0.1
c = 16
T = L

matching = Matching(pcm(L))

fails = []

for i in range(R):
    state = State(N, L)
    decoder_2D(state, T, c, η, p_error)
    x_correction, y_correction = mwpm(matching, state.q)
    fails.append(float(logical_error(x_correction ^ state.x_error, y_correction ^ state.y_error).mean()))
    if time() - t0 > TIMELIMIT:
        break

R = len(fails)
fail_rate = np.array([np.mean(fails), N*R])

if debug:
    print(f"Failure rate: {fail_rate}")
else:
    np.save(f"data/run_11/run_11_{n}.npy", fail_rate)

elapsed = time() - t0
print(f"{N*R} samples for L={L} and p={p_error:.4f} took time:")
print(strftime("%H:%M:%S", gmtime(elapsed)))

used = mem - nvmlDeviceGetMemoryInfo(handle).free / 1000**3
print(f"Actually utilized {used:.2f} GB of VRAM.")

# print(int(elapsed))
