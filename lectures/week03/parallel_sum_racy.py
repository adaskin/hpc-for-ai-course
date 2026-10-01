"""
Week 3 – Race condition demo: shared global accumulator.
Updated with teaching prints for live lecture demonstrations.

Usage:
    python parallel_sum_racy.py
    python parallel_sum_racy.py --runs 3
    python parallel_sum_racy.py --compare
    python parallel_sum_racy.py --safe
"""
import threading
import argparse
import numpy as np
import time

SIZE = 1_000_000
N_WORKERS = 8
N_ADDITIONS_PER_THREAD = 1_000_000

np.random.seed(42)
data = np.random.rand(SIZE)
CHUNK = SIZE // N_WORKERS


def py_add(a, b):
    """Pure-Python add function. 
    A function call acts as a potential GIL switch point, opening a race window.
    """
    return a + b


def run_racy():
    """Racy version: Function calls and assignments inside a shared global loop."""
    global result
    result = 0.0

    def add_chunk(start, end):
        global result
        for i in range(start, end):
            # LOAD global result, LOAD data[i], CALL py_add, STORE global result.
            # The GIL can switch between any of these steps!
            result = py_add(result, data[i])

    threads = []
    for w in range(N_WORKERS):
        s = w * CHUNK
        e = (w + 1) * CHUNK if w < N_WORKERS - 1 else SIZE
        t = threading.Thread(target=add_chunk, args=(s, e))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    return result


def run_racy_numpy_sum():
    """Demonstrates a forced race condition using time.sleep(0).
    time.sleep(0) explicitly yields control, guaranteeing a GIL switch 
    between the LOAD and STORE phases.
    """
    global result
    result = 0.0

    def add_chunk_sum(start, end):
        global result
        for i in range(start, end):
            tmp = result                # 1. LOAD shared state into local variable
            time.sleep(0)               # 2. FORCE YIELD: GIL switches to another thread!
            result = tmp + data[i]      # 3. STORE stale value back, overwriting updates

    threads = []
    for w in range(N_WORKERS):
        s = w * CHUNK
        e = (w + 1) * CHUNK if w < N_WORKERS - 1 else SIZE
        t = threading.Thread(target=add_chunk_sum, args=(s, e))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    return result


def run_safe():
    """Correct version: Each thread computes its chunk locally into private stack memory.
    No shared state during the heavy loop -> zero race conditions.
    """
    results = [0.0] * N_WORKERS

    def add_chunk(idx, start, end):
        local = 0.0
        for i in range(start, end):
            local += data[i]  # Private variable (on thread stack) -> safe!
        results[idx] = local   # Each thread writes to its OWN isolated index.

    threads = []
    for w in range(N_WORKERS):
        s = w * CHUNK
        e = (w + 1) * CHUNK if w < N_WORKERS - 1 else SIZE
        t = threading.Thread(target=add_chunk, args=(w, s, e))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    return sum(results)


def main():
    parser = argparse.ArgumentParser(description="Teaching script for Python GIL and Race Conditions")
    parser.add_argument("--runs", type=int, default=3, help="Number of test runs")
    parser.add_argument("--safe", action="store_true", help="Run the safe local-accumulation version")
    parser.add_argument("--compare", action="store_true", help="Compare with forced-yield / old version")
    args = parser.parse_args()

    expected = float(np.sum(data))
    
    print("=" * 60)
    print("LECTURE DEMO: Python GIL, C-Extensions, and Race Conditions")
    print("=" * 60)
    print(f"Expected Mathematical Sum : {expected:.6f}")
    print(f"Total Workers             : {N_WORKERS} Threads")
    print(f"Dataset Size              : {SIZE:,} elements")
    print()

    if args.compare:
        print("=== 1. FORCED-YIELD VERSION (Simulating a broken shared accumulator) ==.")
        print("Explanation: Using time.sleep(0) forces the GIL to switch right between")
        print("the LOAD and STORE steps, guaranteeing lost updates.")
        print("-" * 60)
        for i in range(3):
            r = run_racy_numpy_sum()
            ok = "✓" if np.isclose(r, expected) else "✗ WRONG (Race Condition!)"
            print(f"  Run {i+1}: {r:.6f}  {ok}")
        print()
        print("  → Teaching Takeaway: Compound operations (LOAD-ADD-STORE) are NOT atomic")
        print("    even with the GIL. Interleaving destroys accuracy.")
        print("=" * 60)
        print()

    print("=== 2. RACY VERSION (Element-by-element global updates) ===")
    print("Explanation: Threads concurrently mutate `result = py_add(result, data[i])`.")
    print(f"{'Run':<6}{'Result':<22}{'Error':<18}{'Status'}")
    print("─" * 60)

    n_correct = 0
    for i in range(args.runs):
        r = run_racy()
        error = abs(r - expected)
        correct = np.isclose(r, expected, rtol=1e-10)
        if correct:
            n_correct += 1
        symbol = "✓ CORRECT" if correct else "✗ WRONG (Racy)"
        print(f"{i+1:<6}{r:<22.6f}{error:<18.6f}{symbol}")

    print("─" * 60)
    print(f"Accuracy Rate: {n_correct}/{args.runs} runs correct.")
    print()

    if args.safe or not args.compare:
        print("=== 3. SAFE VERSION (Local Accumulation Pattern) ===")
        print("Explanation: Threads work entirely on local stack variables. No locks needed.")
        print("-" * 60)
        r_safe = run_safe()
        ok = "✓ CORRECT" if np.isclose(r_safe, expected) else "✗ WRONG"
        print(f"Safe Version Result       : {r_safe:.6f} {ok}")
        print("  → Teaching Takeaway: Avoid shared mutable state. Compute locally, combine safely.")
        print("=" * 60)


if __name__ == "__main__":
    main()