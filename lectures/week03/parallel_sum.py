"""
Week 3 – Parallel sum: performance demo.

Design notes (READ THIS — it's part of the lesson):

1. The task must be COMPUTE-bound, not memory-bandwidth-bound.
   `np.sum(data**2)` on 100M floats is memory-bound and saturates DRAM,
   so splitting across cores gives no speedup even without overhead.

2. Data must be INHERITED via fork, not pickled to workers.
   Pickling 800 MB × 8 workers ≈ 2 s of overhead, which dwarfs the
   ~0.3 s of actual compute. Result: "parallel" is 6× SLOWER.

3. Use fork start method (default on Linux). On macOS/Windows, replace
   with an initializer pattern using multiprocessing.shared_memory.

Usage:
    python parallel_sum.py                    # Scaling demo
    python parallel_sum.py --compare          # Threads vs processes
    python parallel_sum.py --check-correctness
"""
import time
import argparse
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# ── Data (module-level, inherited by forked workers via copy-on-write) ──
SIZE = 20_000_000
np.random.seed(42)
data = np.random.rand(SIZE)

# ── Compute-heavy transform: ~15-20 FLOPs/element ──
# Deliberately NOT memory-bandwidth-bound.
def transform(x):
    return np.sin(x) * np.cos(x) + x ** 2 - np.sqrt(np.abs(x) + 1e-9)

EXPECTED = float(np.sum(transform(data)))

# Use fork so children inherit `data` for free (no pickling of the array!)
CTX = mp.get_context("fork")


def partial_sum(bounds):
    """Worker receives ONLY (start, end) — a few bytes pickled.
    It reads `data` from the parent's address space via fork inheritance."""
    start, end = bounds
    return float(np.sum(transform(data[start:end])))


def make_bounds(n_workers):
    return [(i * SIZE // n_workers, (i + 1) * SIZE // n_workers)
            for i in range(n_workers)]


def run_parallel(n_workers, executor_class):
    kwargs = {}
    if executor_class is ProcessPoolExecutor:
        kwargs["mp_context"] = CTX          # fork, not spawn
    t0 = time.perf_counter()
    with executor_class(max_workers=n_workers, **kwargs) as ex:
        partials = list(ex.map(partial_sum, make_bounds(n_workers)))
    elapsed = time.perf_counter() - t0
    return sum(partials), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--check-correctness", action="store_true")
    args = parser.parse_args()

    workers = [1, 2, 4, 8]

    # ── Mode 1: Scaling (Amdahl's Law) ─────────────────────────────
    if not args.compare and not args.check_correctness:
        print(f"Data size: {SIZE:,} elements")
        print(f"Expected:  {EXPECTED:.6f}")
        print()

        # Serial baseline
        t0 = time.perf_counter()
        serial_result = float(np.sum(transform(data)))
        t_serial = time.perf_counter() - t0

        header = f"{'Workers':<10}{'Time (s)':<12}{'Speedup':<12}{'Efficiency':<12}"
        print(header)
        print("─" * len(header))

        # helper: format value+unit as a single padded cell
        def cell(value_str, width):
            return f"{value_str:<{width}}"

        print(cell("serial", 10)
              + cell(f"{t_serial:.3f}", 12)
              + cell("1.00×", 12)
              + cell("100%", 12))

        for n in workers[1:]:
            result, elapsed = run_parallel(n, ProcessPoolExecutor)
            speedup = t_serial / elapsed
            efficiency = speedup / n * 100
            print(cell(str(n), 10)
                  + cell(f"{elapsed:.3f}", 12)
                  + cell(f"{speedup:.2f}×", 12)
                  + cell(f"{efficiency:.0f}%", 12))

    # ── Mode 2: Threads vs Processes ──────────────────────────────
    elif args.compare:
        print(f"\nThreads vs Processes (data size: {SIZE:,})")
        header = f"{'Workers':<10}{'Processes':<14}{'Threads':<14}{'Note'}"
        print(header)
        print("─" * len(header))
        for n in workers:
            _, t_proc = run_parallel(n, ProcessPoolExecutor)
            _, t_thread = run_parallel(n, ThreadPoolExecutor)
            if n == 1:
                note = ""
            else:
                ratio = t_thread / t_proc          # <1 → threads faster
                if ratio < 0.95:
                    note = f"threads win ({1/ratio:.2f}× faster)"
                elif ratio > 1.05:
                    note = f"processes win ({ratio:.2f}× faster)"
                else:
                    note = "tie"
            print(f"{n:<10}{t_proc:<14.3f}{t_thread:<14.3f}{note}")

    # ── Mode 3: Correctness ────────────────────────────────────────
    elif args.check_correctness:
        print(f"\nExpected: {EXPECTED:.6f}\n")
        for n in [4, 8]:
            r_proc, _ = run_parallel(n, ProcessPoolExecutor)
            r_thread, _ = run_parallel(n, ThreadPoolExecutor)
            ok_p = "✓" if np.isclose(r_proc, EXPECTED) else "✗"
            ok_t = "✓" if np.isclose(r_thread, EXPECTED) else "✗"
            print(f"  {n} workers, Processes: {r_proc:.6f} {ok_p}")
            print(f"  {n} workers, Threads:   {r_thread:.6f} {ok_t}")
        print()
        print("Both correct: workers RETURN results (no shared state).")


if __name__ == "__main__":
    main()