"""Week 6, example 03: element-wise ops — ReLU and sigmoid on 10M elements.

The purest SIMT workload: one thread per element, no communication.
Compare NumPy (vectorised, GIL-releasing) with CuPy on-device.
"""

import time

import numpy as np
import cupy as cp

N = 10_000_000
ITERS = 20


def bench(fn, warmup=None, use_cuda_events=False):
    if warmup is not None:
        warmup()
    if use_cuda_events:
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(ITERS):
            fn()
        end.record()
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end) / ITERS / 1000.0
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    return (time.perf_counter() - t0) / ITERS


if __name__ == "__main__":
    x_cpu = np.random.randn(N).astype(np.float32)
    x_gpu = cp.asarray(x_cpu)

    # --- ReLU: max(0, x) ---
    t_np = bench(lambda: np.maximum(x_cpu, 0, out=x_cpu),
                 warmup=lambda: np.maximum(x_cpu, 0, out=x_cpu))
    t_gpu = bench(lambda: cp.maximum(x_gpu, 0, out=x_gpu),
                  warmup=lambda: cp.maximum(x_gpu, 0, out=x_gpu),
                  use_cuda_events=True)
    print(f"ReLU   10M floats | NumPy {t_np*1e3:7.3f} ms | CuPy {t_gpu*1e3:7.3f} ms | {t_np/t_gpu:5.1f}x")

    # --- Sigmoid: 1 / (1 + exp(-x)) — two ops fused? No: three separate kernels ---
    t_np = bench(lambda: 1.0 / (1.0 + np.exp(-x_cpu)),
                 warmup=lambda: 1.0 / (1.0 + np.exp(-x_cpu)))
    t_gpu = bench(lambda: 1.0 / (1.0 + cp.exp(-x_gpu)),
                  warmup=lambda: 1.0 / (1.0 + cp.exp(-x_gpu)),
                  use_cuda_events=True)
    print(f"Sigmoid 10M floats | NumPy {t_np*1e3:7.3f} ms | CuPy {t_gpu*1e3:7.3f} ms | {t_np/t_gpu:5.1f}x")

    # Bandwidth sanity check: element-wise ops are memory-bound.
    bw = N * 4 / t_gpu / 1e9  # read+write ~ 2x this, order of magnitude is what matters
    print(f"Approximate device bandwidth used: {bw:.0f} GB/s (T4 peak ~320 GB/s)")
