"""
Week 8 – First CUDA kernel: vector addition.
Run on Colab with T4 GPU.
"""
import numpy as np
from numba import cuda

@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

if __name__ == "__main__":
    N = 10_000_000
    a_np = np.random.rand(N).astype(np.float32)
    b_np = np.random.rand(N).astype(np.float32)

    # Upload
    a_d = cuda.to_device(a_np)
    b_d = cuda.to_device(b_np)
    c_d = cuda.device_array(N, dtype=np.float32)

    # Launch config
    tpb = 256
    bpg = (N + tpb - 1) // tpb
    print(f"N={N:,}, threads/block={tpb}, blocks={bpg:,}, total threads={bpg*tpb:,}")

    # Warm up (JIT compile)
    vector_add[bpg, tpb](a_d, b_d, c_d)
    cuda.synchronize()

    # Timed run
    start = cuda.event()
    end = cuda.event()
    start.record()
    for _ in range(10):
        vector_add[bpg, tpb](a_d, b_d, c_d)
    end.record()
    end.synchronize()
    kernel_ms = cuda.event_elapsed_time(start, end) / 10

    # Download and verify
    c_np = c_d.copy_to_host()
    assert np.allclose(c_np, a_np + b_np), "Mismatch!"

    print(f"Kernel time: {kernel_ms:.3f} ms")
    print(f"Bandwidth used: {3 * N * 4 / (kernel_ms / 1000) / 1e9:.0f} GB/s")
    print(f"Correct: ✓")
