"""Week 9, Part VIII: CUDA streams, overlapping copies with compute.

Serial:    upload everything -> run the kernel -> download everything
Streams:   split into chunks; each chunk uploads, computes and downloads in its own
           stream, so chunk k+1's upload overlaps chunk k's kernel.

Overlap needs (1) pinned host memory, (2) asynchronous copies (stream=...) and
(3) independent chunks. The kernel is deliberately arithmetic-heavy so that
compute time is comparable to copy time; otherwise there is nothing to hide.

Run:  python 10_streams_overlap.py [--log2n 24] [--chunks 4] [--iters 200]
"""
import argparse
import time

import numpy as np
from numba import cuda

from gpu_utils import require_gpu

TPB = 256


@cuda.jit
def heavy(x, out, iters):
    i = cuda.grid(1)
    if i < x.size:
        v = x[i]
        for _ in range(iters):
            v = v * 0.999 + 0.5          # converges towards 500: cheap to verify
        out[i] = v


def reference(x, iters):
    v = x.astype(np.float64)
    for _ in range(iters):
        v = v * 0.999 + 0.5
    return v


def run_serial(h_in, h_out, iters):
    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(h_in)
    blocks = (h_in.size + TPB - 1) // TPB
    heavy[blocks, TPB](d_in, d_out, iters)
    d_out.copy_to_host(h_out)
    cuda.synchronize()


def run_streams(h_in, h_out, iters, n_chunks):
    chunk = h_in.size // n_chunks
    streams = [cuda.stream() for _ in range(n_chunks)]
    blocks = (chunk + TPB - 1) // TPB
    for k, stream in enumerate(streams):
        lo, hi = k * chunk, (k + 1) * chunk
        d_in = cuda.to_device(h_in[lo:hi], stream=stream)          # async: host memory is pinned
        d_out = cuda.device_array(chunk, dtype=h_in.dtype, stream=stream)
        heavy[blocks, TPB, stream](d_in, d_out, iters)             # queued behind the upload
        d_out.copy_to_host(h_out[lo:hi], stream=stream)            # queued behind the kernel
    cuda.synchronize()                                             # wait for every stream


def best_ms(fn, n_rep=5):
    fn()                                                           # warm-up (JIT, allocations)
    samples = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log2n", type=int, default=24, help="n = 2**log2n float32 values")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200, help="FMA iterations per element")
    args = ap.parse_args()
    require_gpu()

    n = 1 << args.log2n
    assert n % args.chunks == 0, "chunks must divide n"
    h_in = cuda.pinned_array(n, dtype=np.float32)     # page-locked: required for async copies
    h_out = cuda.pinned_array(n, dtype=np.float32)
    h_in[:] = np.random.default_rng(0).uniform(0, 1000, n).astype(np.float32)
    ref = reference(h_in[: min(n, 1 << 16)], args.iters)

    results = {}
    for name, fn in (("serial", lambda: run_serial(h_in, h_out, args.iters)),
                     (f"{args.chunks} streams", lambda: run_streams(h_in, h_out, args.iters, args.chunks))):
        h_out[:] = 0
        fn()
        assert np.allclose(h_out[: ref.size], ref, rtol=1e-4), f"{name}: wrong result"
        results[name] = best_ms(fn)

    print(f"n = {n:,} float32 ({h_in.nbytes / 2**20:.0f} MiB each way), {args.iters} iterations per element")
    print(f"{'version':<14}{'ms':>10}")
    for name, ms in results.items():
        print(f"{name:<14}{ms:>10.2f}")
    serial, streams = results.values()
    print(f"\nspeedup {serial / streams:.2f}x. Ideal for K balanced stages is 3K / (K + 2) = "
          f"{3 * args.chunks / (args.chunks + 2):.2f}x for K = {args.chunks};")
    print("with unbalanced stages the longest one dominates, and a very short kernel leaves nothing to overlap.")


if __name__ == "__main__":
    main()
