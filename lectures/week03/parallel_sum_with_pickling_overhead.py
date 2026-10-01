"""
Week 3 – Parallel sum of squares: performance demo.
Demonstrates speedup with increasing workers (Amdahl's Law in practice).

Usage:
    python parallel_sum.py              # Run with 1,2,4,8 workers
    python parallel_sum.py --compare    # Compare threads vs processes
    python parallel_sum.py --check-correctness  # Verify answers
"""
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# ── Data ──────────────────────────────────────────────────────────
SIZE = 100_000_000
np.random.seed(42)
data = np.random.rand(SIZE)

EXPECTED = np.sum(data ** 2)  # ground truth


# ── Worker function ───────────────────────────────────────────────
def partial_sum_of_squares(chunk):
    """Each worker computes sum of squares for its chunk."""
    return np.sum(chunk ** 2)


# ── Parallel runner ───────────────────────────────────────────────
def run_parallel(n_workers, executor_class):
    chunks = np.array_split(data, n_workers)
    t0 = time.perf_counter()
    with executor_class(max_workers=n_workers) as ex:
        partials = list(ex.map(partial_sum_of_squares, chunks))
    elapsed = time.perf_counter() - t0
    result = sum(partials)  # combine (serial, tiny)
    return result, elapsed


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--check-correctness", action="store_true")
    args = parser.parse_args()

    workers = [1, 2, 4, 8]

    # ── Mode 1: Scaling demo (Amdahl's Law) ──────────────────────
    if not args.compare and not args.check_correctness:
        print(f"Data size: {SIZE:,} elements")
        print(f"Expected sum of squares: {EXPECTED:.6f}")
        print()

        # Serial baseline
        t0 = time.perf_counter()
        serial_result = np.sum(data ** 2)
        t_serial = time.perf_counter() - t0
        print(f"{'Workers':<10}{'Time (s)':<12}{'Speedup':<10}{'Efficiency':<12}")
        print("─" * 46)
        print(f"{'serial':<10}{t_serial:<12.3f}{'1.00×':<10}{'100%':<12}")

        for n in workers[1:]:  # skip 1 (same as serial)
            result, elapsed = run_parallel(n, ProcessPoolExecutor)
            speedup = t_serial / elapsed
            efficiency = speedup / n * 100
            print(f"{n:<10}{elapsed:<12.3f}{speedup:<10.2f}×{efficiency:<12.0f}%")

    # ── Mode 2: Threads vs Processes ─────────────────────────────
    elif args.compare:
        print(f"\nThreads vs Processes (data size: {SIZE:,})")
        print(f"{'Workers':<10}{'Processes':<14}{'Threads':<14}{'Note'}")
        print("─" * 55)

        for n in workers:
            _, t_proc = run_parallel(n, ProcessPoolExecutor)
            _, t_thread = run_parallel(n, ThreadPoolExecutor)
            note = "" if n == 1 else "← GIL!" if t_thread > t_proc * 1.5 else ""
            print(f"{n:<10}{t_proc:<14.3f}{t_thread:<14.3f}{note}")

    # ── Mode 3: Correctness check ────────────────────────────────
    elif args.check_correctness:
        print(f"\nExpected: {EXPECTED:.6f}")
        print()
        for n in [4, 8]:
            r_proc, _ = run_parallel(n, ProcessPoolExecutor)
            r_thread, _ = run_parallel(n, ThreadPoolExecutor)
            ok_p = "✓" if np.isclose(r_proc, EXPECTED) else "✗"
            ok_t = "✓" if np.isclose(r_thread, EXPECTED) else "✗"
            print(f"  {n} workers, Processes: {r_proc:.6f} {ok_p}")
            print(f"  {n} workers, Threads:   {r_thread:.6f} {ok_t}")
        print()
        print("Both correct because workers RETURN results (no shared state).")


if __name__ == "__main__":
    main()