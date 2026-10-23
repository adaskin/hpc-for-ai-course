"""Week 6, example 06: automatically find the matmul crossover point.

Searches (by doubling) the smallest n where CuPy end-to-end
(upload + matmul + download) beats NumPy on CPU.
Run it twice: results vary with Colab CPU type and current load —
a nice reminder that benchmarking is measurement, not truth.
"""

import time

import numpy as np
import cupy as cp


def numpy_time(n, iters=3):
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    np.dot(a, b)
    t0 = time.perf_counter()
    for _ in range(iters):
        np.dot(a, b)
    return (time.perf_counter() - t0) / iters


def cupy_e2e_time(n):
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    cp.dot(cp.asarray(a), cp.asarray(b))  # warmup (also loads cuBLAS)
    t0 = time.perf_counter()
    ag = cp.asarray(a)
    bg = cp.asarray(b)
    c = cp.dot(ag, bg)
    _ = c.get()
    return time.perf_counter() - t0


if __name__ == "__main__":
    n = 128
    while n <= 16384:
        t_np, t_gpu = numpy_time(n), cupy_e2e_time(n)
        winner = "GPU" if t_gpu < t_np else "CPU"
        print(f"n={n:>6} | NumPy {t_np:8.4f} s | CuPy e2e {t_gpu:8.4f} s | winner: {winner}")
        if t_gpu < t_np:
            print(f"\n=> crossover found: n ≈ {n} "
                  f"(transfer cost at this n: {3*n*n*4/25e9*1e3:.2f} ms over PCIe4)")
            break
        n *= 2
    else:
        print("\n=> GPU never won in this range — are you sure a GPU is attached?")
