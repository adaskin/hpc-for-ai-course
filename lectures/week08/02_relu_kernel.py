"""
Week 8 – Element-wise ReLU on a 2D array (AI-shaped: batch × hidden).
"""
import numpy as np
from numba import cuda

@cuda.jit
def relu_kernel(x, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = max(0.0, x[i])

if __name__ == "__main__":
    batch, hidden = 256, 512
    x_np = np.random.randn(batch, hidden).astype(np.float32)

    x_d = cuda.to_device(x_np.ravel())
    out_d = cuda.device_array(x_np.size, dtype=np.float32)

    tpb = 256
    bpg = (x_np.size + tpb - 1) // tpb
    relu_kernel[bpg, tpb](x_d, out_d)
    cuda.synchronize()

    result = out_d.copy_to_host().reshape(batch, hidden)
    expected = np.maximum(x_np, 0)
    assert np.allclose(result, expected)
    print(f"ReLU on {batch}×{hidden} = {x_np.size:,} elements: ✓")
