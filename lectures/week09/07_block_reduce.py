"""Week 9, Part V: cooperative reduction inside a block, then across blocks.

  shared tree + 1 atomic per block : n/TPB atomics
  one atomic per thread            : n atomics, all queued on one address (Week 3's race, "fixed" by a queue)

Run:  python 07_block_reduce.py [--log2n 24]
"""
import argparse

import numpy as np
from numba import cuda, float32

from gpu_utils import gbs, require_gpu, time_ms

TPB = 256   # threads per block: a power of two, and a compile-time constant for shared.array


@cuda.jit
def block_sums(x, total):
    s = cuda.shared.array(shape=TPB, dtype=float32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    s[tid] = x[i] if i < x.size else 0.0      # pad the tail with the identity of +
    cuda.syncthreads()
    stride = TPB // 2
    while stride > 0:
        if tid < stride:                      # contiguous active lanes: whole warps go idle
            s[tid] += s[tid + stride]
        cuda.syncthreads()                    # OUTSIDE the if: every thread must arrive
        stride //= 2
    if tid == 0:
        cuda.atomic.add(total, 0, s[0])       # one atomic per block


@cuda.jit
def atomic_per_thread(x, total):
    i = cuda.grid(1)
    if i < x.size:
        cuda.atomic.add(total, 0, x[i])       # the Week 3 mistake, on silicon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log2n", type=int, default=24, help="n = 2**log2n float32 values")
    args = ap.parse_args()
    require_gpu()

    n = 1 << args.log2n
    x = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    dx = cuda.to_device(x)
    blocks = (n + TPB - 1) // TPB
    ref = float(x.sum(dtype=np.float64))

    rows = []
    for name, kernel, n_rep, atomics in (("shared tree + 1 atomic", block_sums, 10, blocks),
                                         ("atomic per thread", atomic_per_thread, 3, n)):
        total = cuda.to_device(np.zeros(1, np.float32))
        kernel[blocks, TPB](dx, total)                # one clean run for the result...
        cuda.synchronize()
        result = float(total.copy_to_host()[0])
        scratch = cuda.to_device(np.zeros(1, np.float32))
        ms = time_ms(lambda: kernel[blocks, TPB](dx, scratch), n_rep=n_rep)   # ...timing runs accumulate
        rows.append((name, ms, result, atomics))

    print(f"n = {n:,}   float64 reference sum = {ref:.6f}")
    print(f"{'version':<26}{'ms':>9}{'GB/s':>9}{'atomics':>12}{'result':>14}{'rel err':>11}")
    for name, ms, result, atomics in rows:
        print(f"{name:<26}{ms:>9.3f}{gbs(x.nbytes, ms):>9.1f}{atomics:>12,}{result:>14.4f}"
              f"{abs(result - ref) / abs(ref):>11.2e}")
    print(f"\nThe tree version is {rows[1][1] / rows[0][1]:.1f}x faster: {blocks:,} atomics instead of {n:,}.")
    print("A different summation order gives different last digits; both are 'correct' to float32 precision.")


if __name__ == "__main__":
    main()
