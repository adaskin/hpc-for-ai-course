"""Week 6, example 02: matmul sweep — NumPy (CPU) vs CuPy (GPU).

Times three things at each size n:
  1. NumPy dot        (CPU BLAS, multithreaded — Week 2 callback)
  2. CuPy compute-only (data already resident on GPU; timed with CUDA events)
  3. CuPy end-to-end   (asarray upload + matmul + get() download)

Learning goal: find the crossover point where the GPU starts winning,
and see how PCIe transfer eats the advantage for small n.
"""

import time

import numpy as np

try:
    import cupy as cp
except ImportError:
    raise SystemExit("CuPy not found. On Colab:  pip install cupy-cuda12x")


def time_numpy_matmul(n: int, iters: int = 5) -> float:
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    np.dot(a, b)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        np.dot(a, b)
    return (time.perf_counter() - t0) / iters


def time_cupy_matmul(n: int, iters: int = 20, include_transfer: bool = False) -> float:
    a_cpu = np.random.rand(n, n).astype(np.float32)
    b_cpu = np.random.rand(n, n).astype(np.float32)
    a = cp.asarray(a_cpu)
    b = cp.asarray(b_cpu)
    cp.dot(a, b)  # warmup

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iters):
        c = cp.dot(a, b)
    end.record()
    end.synchronize()
    gpu_only = cp.cuda.get_elapsed_time(start, end) / iters / 1000.0

    if not include_transfer:
        return gpu_only

    t0 = time.perf_counter()
    a2 = cp.asarray(a_cpu)
    b2 = cp.asarray(b_cpu)
    c2 = cp.dot(a2, b2)
    _ = c2.get()
    return time.perf_counter() - t0


if __name__ == "__main__":
    print(f"{'n':>6} | {'NumPy (s)':>10} | {'CuPy GPU (s)':>12} | {'CuPy e2e (s)':>12} | {'speedup':>9}")
    print("-" * 70)
    for n in [256, 512, 1024, 2048, 4096, 8192]:
        t_np = time_numpy_matmul(n)
        t_gpu = time_cupy_matmul(n, include_transfer=False)
        t_e2e = time_cupy_matmul(n, include_transfer=True)
        print(f"{n:>6} | {t_np:>10.4f} | {t_gpu:>12.6f} | {t_e2e:>12.6f} | {t_np / t_gpu:>8.1f}x")
