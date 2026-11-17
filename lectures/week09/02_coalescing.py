"""Week 9, Part III: coalesced vs strided access, measured.

The same matrix, the same arithmetic, two ways to assign threads:
  col_sums: thread j sums column j  -> at step r, lanes j, j+1, ... read A[r, j], A[r, j+1], ...
                                       consecutive addresses: COALESCED
  row_sums: thread i sums row i     -> at step c, lanes i, i+1, ... read A[i, c], A[i+1, c], ...
                                       addresses N floats apart: STRIDED

The matrix is row-major, so "one thread per row" feels natural, and it is the slow one.

Run:  python 02_coalescing.py [--n 4096]
"""

import argparse

import numpy as np
from numba import cuda

from gpu_utils import gbs, require_gpu, time_ms

PEAK_GBS = 320.0  # T4; pass --peak for another GPU
TPB = 256


@cuda.jit
def row_sums(A, out):
    row = cuda.grid(1)
    if row < A.shape[0]:
        acc = 0.0
        for c in range(A.shape[1]):
            acc += A[row, c]  # lane i and lane i+1 are A.shape[1] floats apart
        out[row] = acc


@cuda.jit
def col_sums(A, out):
    col = cuda.grid(1)
    if col < A.shape[1]:
        acc = 0.0
        for r in range(A.shape[0]):
            acc += A[r, col]  # lane i and lane i+1 are adjacent in memory
        out[col] = acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096, help="matrix is n x n float32")
    ap.add_argument(
        "--peak", type=float, default=PEAK_GBS, help="peak bandwidth in GB/s"
    )
    args = ap.parse_args()
    require_gpu()

    n = args.n
    A = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
    dA = cuda.to_device(A)
    d_out = cuda.device_array(n, dtype=np.float32)
    blocks = (n + TPB - 1) // TPB

    print(
        f"{n} x {n} float32 = {A.nbytes / 2**20:.0f} MiB; both kernels read every element once"
    )
    print(f"{'kernel':<28}{'ms':>9}{'GB/s':>9}{'% of peak':>11}")
    for name, kernel, ref in (
        ("col_sums (coalesced)", col_sums, A.sum(axis=0, dtype=np.float64)),
        ("row_sums (strided)", row_sums, A.sum(axis=1, dtype=np.float64)),
    ):
        kernel[blocks, TPB](dA, d_out)
        cuda.synchronize()
        assert np.allclose(
            d_out.copy_to_host(), ref, rtol=1e-4, atol=1e-2
        ), f"{name} is wrong"
        ms = time_ms(lambda: kernel[blocks, TPB](dA, d_out))
        rate = gbs(A.nbytes, ms)
        print(f"{name:<28}{ms:>9.3f}{rate:>9.1f}{100 * rate / args.peak:>10.0f}%")
    print(
        "\nBoth results match NumPy. Same FLOPs, same useful bytes; only the lane-to-address mapping differs."
    )


if __name__ == "__main__":
    # import sys
    # sys.argv = ["prog", "--n", "4096"]
    main()
