"""Week 9, Part IV: shared-memory bank conflicts, on the Week 8 bonus kernel: transpose.

  load   tile[ty, tx]  coalesced global read, conflict-free shared write
  store  tile[tx, ty]  coalesced global write, but a STRIDED shared read:
                       with a (32, 32) tile every lane hits the same bank -> 32-way conflict
  fix    pad the tile to (T, T + 1): consecutive lanes now land in distinct banks

Run:  python 06_bank_conflicts.py [--n 4096]
"""
import argparse

import numpy as np
from numba import cuda, float32

from gpu_utils import gbs, require_gpu, time_ms

T = 32


def make_transpose(pad):
    width = T + (1 if pad else 0)

    @cuda.jit
    def kernel(A, out):
        tile = cuda.shared.array(shape=(T, width), dtype=float32)
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        tile[ty, tx] = A[cuda.blockIdx.y * T + ty, cuda.blockIdx.x * T + tx]
        cuda.syncthreads()
        out[cuda.blockIdx.x * T + ty, cuda.blockIdx.y * T + tx] = tile[tx, ty]

    return kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096, help="matrix is n x n (multiple of 32)")
    args = ap.parse_args()
    require_gpu()

    n = args.n
    assert n % T == 0, "n must be a multiple of 32"
    A = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
    dA = cuda.to_device(A)
    dOut = cuda.device_array((n, n), dtype=np.float32)
    grid, block = (n // T, n // T), (T, T)
    bytes_moved = 2 * A.nbytes      # read A once, write the transpose once

    print(f"transpose {n}x{n} float32, tile {T}x{T}")
    print(f"{'version':<20}{'shared KB':>11}{'ms':>9}{'GB/s':>9}")
    for pad in (False, True):
        kernel = make_transpose(pad)
        kernel[grid, block](dA, dOut)
        cuda.synchronize()
        assert np.array_equal(dOut.copy_to_host(), A.T), "transpose is wrong"
        ms = time_ms(lambda: kernel[grid, block](dA, dOut))
        name = f"padded ({T},{T + 1})" if pad else f"unpadded ({T},{T})"
        shared_kb = T * (T + (1 if pad else 0)) * 4 / 1024
        print(f"{name:<20}{shared_kb:>11.2f}{ms:>9.3f}{gbs(bytes_moved, ms):>9.1f}")

    print("\nBoth versions are correct; padding only changes the bank mapping of the shared read:")
    print(f"  unpadded: bank = (tx*{T} + ty) % 32 = ty          -> 32 lanes, 1 bank    (32-way conflict)")
    print(f"  padded:   bank = (tx*{T + 1} + ty) % 32 = (tx+ty) % 32 -> 32 lanes, 32 banks (no conflict)")
    print("The tiled matmul does not have this problem: its inner reads are broadcasts or conflict-free.")


if __name__ == "__main__":
    main()
