from time import time
from time import strftime, gmtime
t0 = time()
TIMELIMIT = 3600 * 12 # 12 hours
BUFFER = 1800 # 30 minutes
ESTIMATE = 60 # 1 minute

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
n = args.n # 0 - 999
debug = (n == -1)

RUN = 14

if debug:
    L = 50
    p_error = 0.004
else:
    L = int(50 * ((n % 100) // 20 + 1)) # L = 50, 100, 150, 200, 250
    p_error = ((n % 20) + 380) / 100000 # p_error = 0.00380, 0.00381, 0.00382, ..., 0.00399

p_error = cp.float32(p_error)

N = int(2500/(L/100)**2 * mem) # Expect scaling as N L^2 < mem. Constant is empirically determined.
R = TIMELIMIT # Basically repeat until time runs out no matter what
# R = int(10**7/N) # Repetitions, statistical
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
    if i == 1:
        t1 = time()
    state = State(N, L)
    decoder_2D(state, T, c, η, p_error)
    x_correction, y_correction = mwpm(matching, state.q)
    fails.append(float(logical_error(x_correction ^ state.x_error, y_correction ^ state.y_error).mean()))
    if i == 1:
        ESTIMATE = time() - t1
    if time() - t0 + ESTIMATE > TIMELIMIT - BUFFER:
        break

R = len(fails)
fail_rate = np.array([np.mean(fails), N*R])

if fail_rate[0] == 0:
    print("WARNING: No failures detected.")
else:
    print(f"Failure rate: {fail_rate[0]} ± {np.sqrt(fail_rate[0]*(1-fail_rate[0]) / fail_rate[1])}")

if not debug:
    np.save(f"data/run_{RUN}/run_{RUN}_{n}.npy", fail_rate)

elapsed = time() - t0
print(f"{N*R} samples for L={L} and p={p_error:.5f} took time:")
print(strftime("%H:%M:%S", gmtime(elapsed)))

used = mem - nvmlDeviceGetMemoryInfo(handle).free / 1000**3
print(f"Actually utilized {used:.2f} GB of VRAM.")

# print(int(elapsed))
