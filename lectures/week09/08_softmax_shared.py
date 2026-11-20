"""Week 9: redeeming Week 8's naive softmax with a cooperative reduction.

One block per row; three passes over shared memory: max, sum, normalise.
Requires n_cols <= TPB, with TPB a power of two. Padding lanes contribute
-1e30 to the max and 0.0 to the sum.

Run:  python 08_softmax_shared.py [--rows 4096]
"""
import argparse
import math

import numpy as np
from numba import cuda, float32
from scipy.special import softmax as scipy_softmax

from gpu_utils import require_gpu, time_ms

NEG = -1e30   # identity of max for float32 data of ordinary scale


def make_softmax(TPB):
    """Build the kernel for TPB threads per block (a compile-time constant)."""

    @cuda.jit
    def kernel(logits, out):
        s = cuda.shared.array(shape=TPB, dtype=float32)
        row, tid = cuda.blockIdx.x, cuda.threadIdx.x
        n = logits.shape[1]
        val = logits[row, tid] if tid < n else NEG
        s[tid] = val
        cuda.syncthreads()

        stride = TPB // 2                       # pass 1: row max
        while stride > 0:
            if tid < stride:
                s[tid] = max(s[tid], s[tid + stride])
            cuda.syncthreads()
            stride //= 2
        row_max = s[0]                          # every thread reads it...
        cuda.syncthreads()                      # ...before anyone overwrites s

        e = math.exp(val - row_max) if tid < n else 0.0
        s[tid] = e
        cuda.syncthreads()

        stride = TPB // 2                       # pass 2: row sum
        while stride > 0:
            if tid < stride:
                s[tid] += s[tid + stride]
            cuda.syncthreads()
            stride //= 2
        row_sum = s[0]
        cuda.syncthreads()

        if tid < n:                             # pass 3: normalise (e stayed in a register)
            out[row, tid] = e / row_sum

    return kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    args = ap.parse_args()
    require_gpu()

    print(f"{'rows':>6}{'cols':>6}{'TPB':>6}{'ms':>9}{'max err':>11}{'mean row sum':>14}")
    for cols, tpb in ((64, 64), (512, 512), (37, 64)):
        rows = args.rows if cols != 37 else max(args.rows // 4, 1)
        assert cols <= tpb and tpb & (tpb - 1) == 0, "need n_cols <= TPB, TPB a power of two"
        x = (np.random.default_rng(0).standard_normal((rows, cols)) * 5).astype(np.float32)
        dx = cuda.to_device(x)
        dout = cuda.device_array_like(dx)
        kernel = make_softmax(tpb)
        kernel[rows, tpb](dx, dout)
        cuda.synchronize()
        got = dout.copy_to_host()
        ref = scipy_softmax(x, axis=1)
        assert np.allclose(got, ref, atol=1e-5), f"softmax mismatch for {rows}x{cols}"
        ms = time_ms(lambda: kernel[rows, tpb](dx, dout))
        print(f"{rows:>6}{cols:>6}{tpb:>6}{ms:>9.3f}{np.abs(got - ref).max():>11.2e}"
              f"{got.sum(axis=1).mean():>14.6f}")

    print("\nWeek 8: every thread recomputed the whole row. Now: log2(TPB) rounds per reduction.")
    print("The 37-column case exercises the padding path (37 values, 64 threads).")
    print("Wide rows are still slower than cuDNN's softmax, which adds warp shuffles and more tiling.")


if __name__ == "__main__":
    main()
