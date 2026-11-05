"""
Week 8 -- The CUDA C examples (cuda_c/01..07), written in Numba CUDA.

Same index formulas, same launch geometry, same answers; only the spelling changes.
Run on Colab with a T4 GPU:  python 06_indexing_numba.py
(No GPU?  NUMBA_ENABLE_CUDASIM=1 python 06_indexing_numba.py  runs the kernels in a slow
 CPU simulator. Good enough to check the index maths; output order of prints may differ.)
"""
import numpy as np
from numba import cuda


# ---- 01: who am I?  (cuda_c/01_hello_indices.cu) -------------------------------
@cuda.jit
def hello_indices():
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x   # the core formula
    print("block", cuda.blockIdx.x, "thread", cuda.threadIdx.x, "-> i =", i)


# ---- 02: vector add with a bounds check  (cuda_c/02_vector_add_small.cu) ---------
@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)                  # shortcut for blockIdx.x*blockDim.x + threadIdx.x
    if i < c.size:
        c[i] = a[i] + b[i]


# ---- 03: what a missing bounds check does  (cuda_c/03_bounds_check.cu) -----------
@cuda.jit
def fill_safe(out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = i


@cuda.jit
def fill_unsafe(out, n):
    i = cuda.grid(1)
    out[i] = i                        # BUG: threads 10 and 11 write too
    # NOTE: Numba does not bounds-check device arrays. This is as unsafe as CUDA C.


# ---- 06: 2D indexing  (cuda_c/06_matrix_2d.cu) ----------------------------------
@cuda.jit
def fill_matrix(out):
    col, row = cuda.grid(2)           # (x, y) = (column, row), NOT (row, col)!
    if row < out.shape[0] and col < out.shape[1]:
        out[row, col] = 10 * row + col


# ---- 07: grid-stride loop  (cuda_c/07_grid_stride.cu) -----------------------------
@cuda.jit
def who_did_it(owner):
    first = cuda.grid(1)
    stride = cuda.gridsize(1)         # = gridDim.x * blockDim.x
    for i in range(first, owner.size, stride):
        owner[i] = first


@cuda.jit
def saxpy(a, x, y):
    stride = cuda.gridsize(1)
    for i in range(cuda.grid(1), y.size, stride):
        y[i] = a * x[i] + y[i]


if __name__ == "__main__":
    print("== 01 hello indices (3 blocks x 4 threads; line order may vary)")
    hello_indices[3, 4]()
    cuda.synchronize()

    print("\n== 02 vector add, N = 10, 4 threads per block")
    N, tpb = 10, 4
    bpg = (N + tpb - 1) // tpb                              # ceil(10/4) = 3
    a = cuda.to_device(np.arange(N, dtype=np.float32))      # allocate + copy host -> device
    b = cuda.to_device(10 * np.arange(N, dtype=np.float32))
    c = cuda.device_array(N, dtype=np.float32)              # allocate only (uninitialised)
    vector_add[bpg, tpb](a, b, c)
    print(f"blocks = {bpg}, launched threads = {bpg * tpb}, idle = {bpg * tpb - N}")
    print("c =", c.copy_to_host().astype(int))              # copy_to_host waits for the kernel

    print("\n== 03 bounds check (12 cells, n = 10, guard cells hold -1)")
    for kernel in (fill_safe, fill_unsafe):
        d = cuda.to_device(np.full(12, -1, dtype=np.int32))
        kernel[3, 4](d, 10)
        print(f"{kernel.__name__:12s}", d.copy_to_host())

    print("\n== 06 2D indexing, 5 x 7 matrix, 4 x 4 threads per block")
    rows, cols = 5, 7
    threads = (4, 4)                                        # (x = columns, y = rows)
    blocks = ((cols + 3) // 4, (rows + 3) // 4)             # (2, 2)
    m = cuda.to_device(np.full((rows, cols), -1, dtype=np.int32))
    fill_matrix[blocks, threads](m)
    print(m.copy_to_host())

    print("\n== 07 grid-stride loop, N = 20, 2 blocks x 4 threads = 8 threads")
    owner = cuda.to_device(np.full(20, -1, dtype=np.int32))
    who_did_it[2, 4](owner)
    print("element:", np.arange(20))
    print("thread :", owner.copy_to_host())

    print("\n== 07 saxpy, same 32 x 256 launch for every N")
    for n in (1000, 1 << 20, 5_000_000):
        x = cuda.to_device(np.ones(n, dtype=np.float32))
        y = cuda.to_device(np.full(n, 2, dtype=np.float32))
        saxpy[32, 256](np.float32(2.0), x, y)
        wrong = int((y.copy_to_host() != 4.0).sum())
        print(f"N = {n:8d}: wrong elements = {wrong}")
