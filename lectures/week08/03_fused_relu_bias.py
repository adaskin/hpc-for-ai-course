"""
Week 8 – Kernel fusion: ReLU + bias add in one kernel vs two separate kernels.
Shows the memory-traffic benefit of fusion.
"""
import numpy as np
from numba import cuda

@cuda.jit
def relu_only(x, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = max(0.0, x[i])

@cuda.jit
def bias_add(x, bias, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = x[i] + bias[i % bias.size]

@cuda.jit
def fused_relu_bias(x, bias, out):
    """One kernel: ReLU + bias. One read, one write."""
    i = cuda.grid(1)
    if i < x.size:
        out[i] = max(0.0, x[i]) + bias[i % bias.size]

if __name__ == "__main__":
    N = 10_000_000
    hidden = 512
    x_np = np.random.randn(N).astype(np.float32)
    bias_np = np.random.randn(hidden).astype(np.float32)

    x_d = cuda.to_device(x_np)
    bias_d = cuda.to_device(bias_np)
    out_d = cuda.device_array(N, dtype=np.float32)
    tmp_d = cuda.device_array(N, dtype=np.float32)

    tpb = 256
    bpg = (N + tpb - 1) // tpb

    # Warm up
    relu_only[bpg, tpb](x_d, tmp_d)
    bias_add[bpg, tpb](tmp_d, bias_d, out_d)
    fused_relu_bias[bpg, tpb](x_d, bias_d, out_d)
    cuda.synchronize()

    # Time unfused (2 kernels)
    start, end = cuda.event(), cuda.event()
    start.record()
    for _ in range(10):
        relu_only[bpg, tpb](x_d, tmp_d)
        bias_add[bpg, tpb](tmp_d, bias_d, out_d)
    end.record(); end.synchronize()
    t_unfused = cuda.event_elapsed_time(start, end) / 10

    # Time fused (1 kernel)
    start.record()
    for _ in range(10):
        fused_relu_bias[bpg, tpb](x_d, bias_d, out_d)
    end.record(); end.synchronize()
    t_fused = cuda.event_elapsed_time(start, end) / 10

    # Verify the fused result, including the non-whole number of bias tiles.
    fused_relu_bias[bpg, tpb](x_d, bias_d, out_d)
    result = out_d.copy_to_host()
    bias_tiled = np.tile(bias_np, (N + hidden - 1) // hidden)[:N]
    expected = np.maximum(x_np, 0) + bias_tiled
    assert np.allclose(result, expected, atol=1e-5)

    print(f"Unfused (2 kernels): {t_unfused:.3f} ms")
    print(f"Fused   (1 kernel):  {t_fused:.3f} ms")
    print(f"Speedup from fusion: {t_unfused/t_fused:.2f}×")
