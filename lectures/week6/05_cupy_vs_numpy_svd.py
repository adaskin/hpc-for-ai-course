"""Week 6, example 05: SVD — NumPy (LAPACK) vs CuPy (cuSOLVER).

Callback to Project 1 (HP-SS engine): the same linear algebra,
different backend. Good example of a workload that is NOT purely
data-parallel (reduction-heavy), so the speedup is more modest.
"""

import time

import numpy as np
import cupy as cp

if __name__ == "__main__":
    print(f"{'n':>6} | {'NumPy (s)':>10} | {'CuPy (s)':>10} | {'speedup':>8}")
    print("-" * 44)
    for n in [1000, 2000, 4000]:
        a = np.random.rand(n, n).astype(np.float32)

        t0 = time.perf_counter()
        np.linalg.svd(a)
        t_np = time.perf_counter() - t0

        a_gpu = cp.asarray(a)
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        cp.linalg.svd(a_gpu)
        end.record()
        end.synchronize()
        t_gpu = cp.cuda.get_elapsed_time(start, end) / 1000.0

        print(f"{n:>6} | {t_np:>10.3f} | {t_gpu:>10.3f} | {t_np/t_gpu:>7.1f}x")

# Discussion: why is the speedup smaller than for matmul?
#   - SVD is a sequence of dependent factorisation steps, not one big GEMM.
#   - Wavefront/parallelism is lower; cuSOLVER kernels are less "wide".
#   - Moral: GPU speedups are workload-dependent — always measure (Week 1!).
