"""
Week 2 – Benchmark: Python vs NumPy vs Numba vs Numba parallel.
Run: python benchmark_all.py
"""
import time
import numpy as np
from numba import njit, prange


# ── Python (slow) ─────────────────────────────────────────────────

def pairwise_distances_python(A, B):
    n, d = A.shape
    m, _ = B.shape
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            total = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                total += diff * diff
            D[i, j] = total ** 0.5
    return D


# ── NumPy (matmul trick) ──────────────────────────────────────────

def pairwise_distances_numpy(A, B):
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)
    B_sq = np.sum(B ** 2, axis=1, keepdims=True)
    cross = A @ B.T
    dist_sq = A_sq + B_sq.T - 2 * cross
    return np.sqrt(np.maximum(dist_sq, 0))


# ── Numba serial ──────────────────────────────────────────────────

@njit
def pairwise_distances_numba(A, B):
    n = A.shape[0]
    m = B.shape[0]
    d = A.shape[1]
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            total = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                total += diff * diff
            D[i, j] = total ** 0.5
    return D


# ── Numba parallel ────────────────────────────────────────────────

@njit(parallel=True)
def pairwise_distances_parallel(A, B):
    n = A.shape[0]
    m = B.shape[0]
    d = A.shape[1]
    D = np.zeros((n, m))
    for i in prange(n):
        for j in range(m):
            total = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                total += diff * diff
            D[i, j] = total ** 0.5
    return D


# ── Benchmark harness ─────────────────────────────────────────────

def bench(fn, *args, n_runs=5, warmup=True):
    if warmup:
        fn(*args)  # warm-up / compile
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)


def main():
    np.random.seed(42)
    A = np.random.rand(200, 50)
    B = np.random.rand(500, 50)

    print("Warming up Numba (first compilation)...")
    pairwise_distances_numba(A, B)
    pairwise_distances_parallel(A, B)
    print("Done.\n")

    print("=" * 65)
    print(f"{'Version':<30} {'Time (ms)':>12} {'Speedup':>12}")
    print("=" * 65)

    t_py = bench(pairwise_distances_python, A, B, warmup=False) * 1000
    print(f"{'Python loop':<30} {t_py:>12.2f} {'1×':>12}")

    t_np = bench(pairwise_distances_numpy, A, B) * 1000
    print(f"{'NumPy (matmul trick)':<30} {t_np:>12.4f} {t_py/t_np:>11.0f}×")

    t_nb = bench(pairwise_distances_numba, A, B) * 1000
    print(f"{'Numba @njit (serial)':<30} {t_nb:>12.4f} {t_py/t_nb:>11.0f}×")

    t_par = bench(pairwise_distances_parallel, A, B) * 1000
    print(f"{'Numba parallel (prange)':<30} {t_par:>12.4f} {t_py/t_par:>11.0f}×")

    print("=" * 65)


if __name__ == "__main__":
    main()

