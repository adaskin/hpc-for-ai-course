"""
Week 8 – Row-wise softmax (one block per row).
Naive: assumes row length <= threads per block.
Every thread redundantly computes the row max and sum; no shared memory yet.
"""
import math
import numpy as np
from numba import cuda

@cuda.jit
def softmax_kernel(logits, out):
    """One block per row; intentionally duplicated scalar reductions."""
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n = logits.shape[1]

    if row >= logits.shape[0] or tid >= n:
        return

    val = logits[row, tid]

    # Step 1: find max (every thread repeats this loop by design)
    row_max = val
    for j in range(n):
        if logits[row, j] > row_max:
            row_max = logits[row, j]

    # Step 2: exp and sum
    exp_val = math.exp(val - row_max)
    row_sum = 0.0
    for j in range(n):
        row_sum += math.exp(logits[row, j] - row_max)

    # Step 3: normalise
    out[row, tid] = exp_val / row_sum

if __name__ == "__main__":
    batch, seq_len = 32, 64
    logits_np = np.random.randn(batch, seq_len).astype(np.float32)

    logits_d = cuda.to_device(logits_np)
    out_d = cuda.device_array((batch, seq_len), dtype=np.float32)

    softmax_kernel[batch, seq_len](logits_d, out_d)
    cuda.synchronize()

    result = out_d.copy_to_host()

    # Verify against scipy
    from scipy.special import softmax as sp_softmax
    expected = sp_softmax(logits_np, axis=1)
    assert np.allclose(result, expected, atol=1e-5)
    print(f"Softmax on {batch}×{seq_len}: ✓")
