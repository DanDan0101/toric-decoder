from numba import cuda
from numpy import ceil
import cupy as cp
import subprocess as sp

@cuda.jit(device = True)
def argmax(values):
    """
    Helper device function to find the index of the maximum value in a list.

    Parameters:
    values (Iterable): The list of values to search.

    Returns:
    int: The index of the maximum value.
    """
    max_arg = 0
    max_value = values[0]
    for i in range(1, len(values)):
        value = values[i]
        if value > max_value:
            max_value = value
            max_arg = i
    return max_arg

@cuda.jit
def neighbor_argmax_kernel(a, out):
    """
    CUDA kernel to find the index of the neighbor with the largest value in a 3D array, along axes 1 and 2.

    Parameters:
    a (cp.ndarray): An N x L x L array to search, ∈ ℝ.
    out (cp.ndarray): An N x L x L array to store the results, ∈ {0, 1, 2, 3}.
    """
    d, x, y = cuda.grid(3)
    N, L, _ = out.shape
    if d < N and x < L and y < L:
        out[d, x, y] = argmax((
            a[d, (x-1)%L, y      ],
            a[d, x      , (y-1)%L],
            a[d, x      , (y+1)%L],
            a[d, (x+1)%L, y      ]
        ))

def neighbor_argmax(ϕ):
    """
    Wrapper for neighbor_argmax_kernel which invokes the CUDA kernel.

    Parameters:
    ϕ (cp.ndarray): An N x L x L array to search, ∈ ℝ.

    Returns:
    cp.ndarray: An N x L x L array containing the indices of the neighbors with the largest value.
    """
    shape = ϕ.shape
    out = cp.empty(shape, dtype = cp.uint8)
    threadsperblock = (32, 4, 4) # 512 threads per block
    blockspergrid = tuple(int(ceil(shape[i] / threadsperblock[i])) for i in range(3))
    neighbor_argmax_kernel[blockspergrid, threadsperblock](ϕ, out)
    cuda.synchronize() # kernel is asynchronous
    return out

def get_gpu_memory():
    """
    Query the current GPU memory usage from nvidia-smi.

    Returns:
    int: The memory used in MiB.
    """
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_values = [int(x.split()[0]) for x in memory_used_info]
    return memory_used_values[0]