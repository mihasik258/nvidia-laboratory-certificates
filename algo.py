import numpy as np
from numba import cuda
import math
import time

rows, cols = 4096, 4096
n = rows * cols

x = np.linspace(0, 6 * np.pi, cols, dtype=np.float32)
y = np.linspace(0, 4 * np.pi, rows, dtype=np.float32)
xx, yy = np.meshgrid(x, y)

a = (
    np.sin(xx) * np.cos(yy) * 50 +
    np.random.normal(0, 5, (rows, cols)).astype(np.float32) +
    (yy / (4 * np.pi)) * 100
).astype(np.float32)

start_cpu = time.perf_counter()
sorted_cpu = np.sort(a, axis=1)
end_cpu = time.perf_counter()
cpu_time = end_cpu - start_cpu
print(f"CPU sorting time: {cpu_time:.4f} seconds")

d_a = cuda.to_device(a)
@cuda.jit
def row_bitonic_kernel(d_arr, k, j):
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row >= d_arr.shape[0] or col >= d_arr.shape[1]:
        return

    i = col
    ixj = i ^ j

    if ixj > i:
        asc = (i & k) == 0
        if (asc and d_arr[row, i] > d_arr[row, ixj]) or (not asc and d_arr[row, i] < d_arr[row, ixj]):
            tmp = d_arr[row, i]
            d_arr[row, i] = d_arr[row, ixj]
            d_arr[row, ixj] = tmp


tpb_x, tpb_y = 1024, 1
blocks_x = math.ceil(cols / tpb_x)
blocks_y = math.ceil(rows / tpb_y)
threads_per_block = (tpb_x, tpb_y)
blocks = (blocks_x, blocks_y)
log_cols = int(math.log2(cols))

for stage in range(1, log_cols + 1):
    k = 1 << stage
    for step in range(stage, 0, -1):
        j = 1 << (step - 1)
        row_bitonic_kernel[blocks, threads_per_block](d_a, k, j)
cuda.synchronize()
d_a = cuda.to_device(a)
start_gpu = time.perf_counter()

for stage in range(1, log_cols + 1):
    k = 1 << stage
    for step in range(stage, 0, -1):
        j = 1 << (step - 1)
        row_bitonic_kernel[blocks, threads_per_block](d_a, k, j)

cuda.synchronize()
end_gpu = time.perf_counter()
gpu_time = end_gpu - start_gpu
print(f"GPU sorting time: {gpu_time:.4f} seconds")

sorted_gpu = d_a.copy_to_host()
if np.allclose(sorted_cpu, sorted_gpu, atol=1e-3):
    print("Sorting results match.")
else:
    print("Sorting results do not match.")
