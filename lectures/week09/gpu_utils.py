"""Week 9 helpers: kernel timing with CUDA events, achieved bandwidth, resource report.

Imported by the other scripts in this directory (run them from here, or as
`python week09/NN_name.py` from the repository root).

Timing notes
    - Warm up first: the first launch includes JIT compilation.
    - Kernel launches are asynchronous, so record an end event and synchronize on it.
    - `cuda.event()` records timing by default (`cuda.event(timing=False)` would disable it).
    - Report a median over several runs, not a single sample.
"""
import numpy as np
from numba import cuda


def require_gpu():
    """Stop with a clear message when no CUDA device (or simulator) is available."""
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU found. On Colab: Runtime -> Change runtime type -> T4 GPU. "
                         "No GPU? NUMBA_ENABLE_CUDASIM=1 runs small sizes on the CPU simulator "
                         "(correctness only, timings are meaningless).")


def time_ms(fn, n_warm=3, n_rep=10):
    """Median time of `fn()` in milliseconds, measured with CUDA events."""
    for _ in range(n_warm):
        fn()
    cuda.synchronize()
    samples = []
    for _ in range(n_rep):
        start, end = cuda.event(), cuda.event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return float(np.median(samples))


def gbs(bytes_moved, ms):
    """Achieved bandwidth in GB/s."""
    return bytes_moved / (ms * 1e-3) / 1e9


def gflops(flops, ms):
    """Achieved throughput in GFLOP/s."""
    return flops / (ms * 1e-3) / 1e9


def device_report():
    """Print the device limits that matter for occupancy."""
    dev = cuda.get_current_device()
    name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
    print(f"device: {name}")
    for attr in ("MULTIPROCESSOR_COUNT", "WARP_SIZE", "MAX_THREADS_PER_BLOCK",
                 "MAX_THREADS_PER_MULTIPROCESSOR", "MAX_BLOCKS_PER_MULTIPROCESSOR",
                 "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR", "MAX_REGISTERS_PER_MULTIPROCESSOR"):
        print(f"  {attr:40s} {getattr(dev, attr, 'n/a')}")
    return dev


def regs_per_thread(kernel):
    """Registers per thread of a compiled, specialised kernel, or 0 if unknown."""
    try:
        regs = kernel.get_regs_per_thread()
        return int(max(regs.values())) if isinstance(regs, dict) else int(regs)
    except Exception:
        return 0


def kernel_resources(kernel):
    """Print registers per thread and local (spill) bytes per thread of a compiled kernel.

    Call it after the kernel has been launched at least once, so that it is compiled.
    """
    try:
        regs = kernel.get_regs_per_thread()
        local = kernel.get_local_mem_per_thread()
        for sig in regs:
            print(f"  compiled for {sig}: {regs[sig]} regs/thread, "
                  f"{local[sig]} local (spill) bytes/thread")
    except Exception as exc:  # e.g. CUDA simulator, or an older Numba
        print(f"  (resource report unavailable: {exc}; use ncu or the occupancy calculator)")


def theoretical_occupancy(threads_per_block, shared_bytes, regs_per_thread, dev=None):
    """Blocks per SM and theoretical occupancy (%), from the per-SM resource limits.

    Ignores register-allocation granularity, so it is a slightly optimistic estimate.
    Returns (blocks_per_sm, occupancy_percent, blocks_allowed_by_each_resource).
    """
    dev = dev or cuda.get_current_device()
    max_threads = getattr(dev, "MAX_THREADS_PER_MULTIPROCESSOR", 1024)
    max_blocks = getattr(dev, "MAX_BLOCKS_PER_MULTIPROCESSOR", 16)
    shared_sm = getattr(dev, "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR", 65536)
    regs_sm = getattr(dev, "MAX_REGISTERS_PER_MULTIPROCESSOR", 65536)
    by_threads = max_threads // threads_per_block
    by_shared = shared_sm // shared_bytes if shared_bytes else max_blocks
    by_regs = regs_sm // (regs_per_thread * threads_per_block) if regs_per_thread else max_blocks
    blocks = max(1, min(by_threads, max_blocks, by_shared, by_regs))
    occupancy = 100.0 * blocks * threads_per_block / max_threads
    return blocks, occupancy, dict(threads=by_threads, blocks=max_blocks,
                                   shared=by_shared, registers=by_regs)
