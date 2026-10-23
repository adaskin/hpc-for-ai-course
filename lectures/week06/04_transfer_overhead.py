"""Week 6, example 04: the transfer trap — when PCIe dominates.

Times cp.asarray (CPU->GPU upload) alone as a function of array size,
then pits a tiny upload against a tiny compute to show the round-trip
latency floor (~10-50 us), which makes the GPU SLOWER than the CPU
for small data — the granularity question of Week 5 applied to PCIe.
"""

import time

import numpy as np
import cupy as cp

if __name__ == "__main__":
    print("Upload bandwidth vs size (fixed round-trip latency):")
    print(f"{'MB':>8} | {'upload time':>12} | {'effective GB/s':>14}")
    print("-" * 42)
    for mb in [0.004, 0.4, 4, 40, 400]:
        n = int(mb * 1e6 / 4)
        x = np.random.rand(n).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(20):
            x_gpu = cp.asarray(x)
        cp.cuda.Stream.null.synchronize()
        t_up = (time.perf_counter() - t0) / 20
        print(f"{mb:>8.3f} | {t_up*1e6:>10.1f} us | {n*4/t_up/1e9:>14.2f}")

    # The trap: a tiny computation on the GPU, done via PCIe round-trips.
    small = np.random.rand(1_000).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(100):
        _ = np.maximum(small, 0)          # CPU, everything in cache
    t_cpu = (time.perf_counter() - t0) / 100

    t0 = time.perf_counter()
    for _ in range(100):
        _ = cp.maximum(cp.asarray(small), 0).get()   # upload + kernel + download
    cp.cuda.Stream.null.synchronize()
    t_gpu = (time.perf_counter() - t0) / 100

    print(f"\n1k-element ReLU: CPU {t_cpu*1e6:6.1f} us | GPU with transfers {t_gpu*1e6:6.1f} us"
          f"  -> GPU is {t_gpu/t_cpu:.1f}x SLOWER here")
