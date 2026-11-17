"""Week 9, Part I: the naive matmul, timed and compared with the traffic model.

One thread per element of C. Each thread reads a full row of A and a full column
of B from global memory, so every element of A and B is fetched N times.

Run:  python 03_matmul_naive.py [--n 1024]
"""

import argparse

import numpy as np
from numba import cuda

from gpu_utils import gbs, gflops, require_gpu, time_ms

BLOCK = 16  # threads per block side


@cuda.jit
def matmul_naive(A, B, C):
    col, row = cuda.grid(2)  # x -> columns, y -> rows (Week 8 convention)
    if row < C.shape[0] and col < C.shape[1]:
        acc = 0.0
        for k in range(A.shape[1]):
            acc += (
                A[row, k] * B[k, col]
            )  # both operands come from VRAM, every iteration
        C[row, col] = acc


def run(n):
    """Return (median ms, max abs error vs NumPy) for an n x n float32 matmul."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    dA, dB = cuda.to_device(A), cuda.to_device(B)
    dC = cuda.device_array((n, n), dtype=np.float32)
    grid = ((n + BLOCK - 1) // BLOCK, (n + BLOCK - 1) // BLOCK)
    matmul_naive[grid, (BLOCK, BLOCK)](dA, dB, dC)
    cuda.synchronize()
    err = float(np.abs(dC.copy_to_host() - A @ B).max())
    ms = time_ms(lambda: matmul_naive[grid, (BLOCK, BLOCK)](dA, dB, dC))
    return ms, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()
    require_gpu()

    n = args.n
    ms, err = run(n)
    model_bytes = 2 * n**3 * 4  # no-reuse model from 01_traffic_accounting.py
    print(
        f"naive matmul N={n}: {ms:.3f} ms  {gflops(2 * n**3, ms):.1f} GFLOP/s  max err {err:.2e}"
    )
    print(
        f"  no-reuse model moves {model_bytes / 1e9:.2f} GB from VRAM "
        f"-> {model_bytes / 320e9 * 1e3:.1f} ms at 320 GB/s"
    )
    print(
        f"  effective rate if the model were exact: {gbs(model_bytes, ms):.0f} GB/s "
        "(above 320 means L1/L2 caches served part of the reads)"
    )


if __name__ == "__main__":
    main()
