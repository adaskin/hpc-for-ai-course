"""
Week 8 – Benchmark: your kernel vs CuPy vs NumPy.
Compute-only comparison with preallocated outputs and warm-up.
"""
import time
import numpy as np
import cupy as cp
from numba import cuda

@cuda.jit
def vector_add_kernel(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

def bench_numba(fn, n_runs=10):
    times = []
    for _ in range(n_runs):
        start, end = cuda.event(), cuda.event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(cuda.event_elapsed_time(start, end))
    return float(np.median(times))

def bench_cupy(fn, n_runs=10):
    times = []
    for _ in range(n_runs):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))
    return float(np.median(times))

if __name__ == "__main__":
    N = 10_000_000
    a_np = np.random.rand(N).astype(np.float32)
    b_np = np.random.rand(N).astype(np.float32)
    out_np = np.empty_like(a_np)

    # NumPy (CPU), preallocated output
    np.add(a_np, b_np, out=out_np)  # warm up
    t0 = time.perf_counter()
    for _ in range(5):
        np.add(a_np, b_np, out=out_np)
    t_numpy = (time.perf_counter() - t0) / 5 * 1000

    # CuPy, preallocated output
    a_cp, b_cp = cp.asarray(a_np), cp.asarray(b_np)
    out_cp = cp.empty_like(a_cp)
    cp.add(a_cp, b_cp, out=out_cp)  # warm up
    cp.cuda.Stream.null.synchronize()
    t_cupy = bench_cupy(lambda: cp.add(a_cp, b_cp, out=out_cp))

    # Numba CUDA, preallocated output
    a_d, b_d = cuda.to_device(a_np), cuda.to_device(b_np)
    c_d = cuda.device_array(N, dtype=np.float32)
    tpb, bpg = 256, (N + 255) // 256
    vector_add_kernel[bpg, tpb](a_d, b_d, c_d)  # JIT + warm up
    cuda.synchronize()
    t_numba = bench_numba(lambda: vector_add_kernel[bpg, tpb](a_d, b_d, c_d))

    assert np.allclose(c_d.copy_to_host(), a_np + b_np)
    print(f"N = {N:,}")
    print(f"NumPy (CPU):     {t_numpy:.3f} ms")
    print(f"CuPy:            {t_cupy:.3f} ms")
    print(f"Numba CUDA:      {t_numba:.3f} ms")
    print(f"Numba vs CuPy:   {t_cupy/t_numba:.2f}× (CuPy time / Numba time)")
