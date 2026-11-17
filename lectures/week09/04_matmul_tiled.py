"""Week 9, Part IV: shared-memory tiling, and what happens when a barrier is removed.

Run:  python 04_matmul_tiled.py                 # correct kernel + verification
      python 04_matmul_tiled.py --race          # also run the two broken variants
      python 04_matmul_tiled.py --n 256 --tile 16

Numba note: cuda.shared.array needs a COMPILE-TIME CONSTANT shape, so the kernel is
built by a factory function that closes over T.  N must be a multiple of T (no bounds checks).
"""
import argparse

import numpy as np
from numba import cuda, float32

from gpu_utils import (gflops, kernel_resources, regs_per_thread, require_gpu,
                       theoretical_occupancy, time_ms)


def make_tiled(T, drop_barrier=None):
    """Build a tiled matmul kernel for T x T tiles.

    drop_barrier: None (correct), "after_load" or "after_compute" (broken, for the race demo).
    """
    sync_after_load = drop_barrier != "after_load"
    sync_after_compute = drop_barrier != "after_compute"

    @cuda.jit
    def kernel(A, B, C):
        sA = cuda.shared.array(shape=(T, T), dtype=float32)
        sB = cuda.shared.array(shape=(T, T), dtype=float32)
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        bx, by = cuda.blockIdx.x, cuda.blockIdx.y
        row, col = by * T + ty, bx * T + tx
        acc = 0.0                                   # a register, private to this thread
        for k0 in range(0, A.shape[1], T):
            sA[ty, tx] = A[row, k0 + tx]            # one element of each tile per thread
            sB[ty, tx] = B[k0 + ty, col]
            if sync_after_load:
                cuda.syncthreads()                  # (1) the tile is complete
            for k in range(T):
                acc += sA[ty, k] * sB[k, tx]        # T reuses of every loaded value
            if sync_after_compute:
                cuda.syncthreads()                  # (2) nobody overwrites the tile early
        C[row, col] = acc

    return kernel


def run(n, T, drop=None):
    """Return (median ms, max abs error vs NumPy, compiled kernel)."""
    assert n % T == 0, "n must be a multiple of the tile size"
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    dA, dB = cuda.to_device(A), cuda.to_device(B)
    dC = cuda.device_array((n, n), dtype=np.float32)
    kernel = make_tiled(T, drop)
    grid, block = (n // T, n // T), (T, T)
    kernel[grid, block](dA, dB, dC)
    cuda.synchronize()
    err = float(np.abs(dC.copy_to_host() - A @ B).max())
    ms = time_ms(lambda: kernel[grid, block](dA, dB, dC))
    return ms, err, kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--tile", type=int, default=16, choices=(8, 16, 32))
    ap.add_argument("--race", action="store_true", help="also run the barrier-removed variants")
    args = ap.parse_args()
    require_gpu()

    n, T = args.n, args.tile
    ms, err, kernel = run(n, T)
    print(f"tiled N={n} T={T}: {ms:.3f} ms  {gflops(2 * n**3, ms):.1f} GFLOP/s  max err {err:.2e}")
    assert err < 1e-2, "the tiled kernel disagrees with NumPy"
    kernel_resources(kernel)
    blocks, occ, limits = theoretical_occupancy(T * T, 2 * T * T * 4, regs_per_thread(kernel))
    print(f"  theoretical occupancy: {blocks} blocks/SM -> {occ:.0f}%   (limits: {limits})")

    try:
        import cupy as cp
        z = cp.zeros((n, n), cp.float32)
        t_cublas = time_ms(lambda: z @ z)
        print(f"cuBLAS via CuPy: {t_cublas:.3f} ms -> our kernel is {ms / t_cublas:.1f}x slower")
    except Exception as exc:
        print(f"(CuPy comparison skipped: {exc})")

    if args.race:
        print("\nRemoving a barrier (run each a few times: the answer may vary):")
        for drop in ("after_load", "after_compute"):
            for rep in range(3):
                _, e, _ = run(n, T, drop)
                verdict = "WRONG" if e > 1e-2 else "looked right this time"
                print(f"  drop={drop:<14} run {rep}: max err {e:.3e}   {verdict}")
        print("A race is not a crash: sometimes the answer looks correct. That is what makes it dangerous.")


if __name__ == "__main__":
    main()
