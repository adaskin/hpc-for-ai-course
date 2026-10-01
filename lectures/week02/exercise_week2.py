"""
Week 2 – Student Exercise: Make it 100× faster.

You are given three slow functions from Week 1.
Your task:
  1. Vectorise each with NumPy.
  2. JIT-compile pairwise_distances with Numba (@njit, then parallel).
  3. Benchmark all versions.

Fill in the TODO sections. Run the script and compare times.
"""
import time
import numpy as np
from numba import njit, prange


# ── Original slow functions (from Week 1) ─────────────────────────

def normalize_rows_python(matrix):
    result = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        norm = 0.0
        for j in range(matrix.shape[1]):
            norm += matrix[i, j] ** 2
        norm = norm ** 0.5
        for j in range(matrix.shape[1]):
            result[i, j] = matrix[i, j] / norm
    return result


def softmax_python(logits):
    result = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        max_val = logits[i, 0]
        for j in range(1, logits.shape[1]):
            if logits[i, j] > max_val:
                max_val = logits[i, j]
        total = 0.0
        for j in range(logits.shape[1]):
            result[i, j] = np.exp(logits[i, j] - max_val)
            total += result[i, j]
        for j in range(logits.shape[1]):
            result[i, j] /= total
    return result


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


# ── TODO 1: NumPy vectorised versions ─────────────────────────────

def normalize_rows_numpy(matrix):
    """Vectorise using np.linalg.norm and broadcasting."""
    # TODO: your code here (2 lines)
    pass


def softmax_numpy(logits):
    """Vectorise using np.exp, np.max, broadcasting."""
    # TODO: your code here (3 lines)
    pass


def pairwise_distances_numpy(A, B):
    """Use the algebraic expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b"""
    # TODO: your code here (5 lines)
    pass


# ── TODO 2: Numba versions of pairwise_distances ──────────────────

@njit
def pairwise_distances_numba(A, B):
    """Same triple loop as Python version. Just add the decorator."""
    # TODO: copy the body from pairwise_distances_python
    pass


@njit(parallel=True)
def pairwise_distances_parallel(A, B):
    """Parallel version: use prange for the outer loop."""
    # TODO: copy body, change range(n) to prange(n)
    pass


# ── Benchmark harness ─────────────────────────────────────────────

def bench(fn, *args, n_runs=5):
    fn(*args)  # warm-up
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)


def main():
    np.random.seed(42)
    X = np.random.rand(2000, 50)
    Y = np.random.rand(500, 50)
    logits = np.random.rand(2000, 10)
    X_sub = X[:200]  # subset for manageable Python-loop runtime

    print("Benchmarking... (Python loops may take a while)\n")
    print(f"{'Function':<22} {'Python':>10} {'NumPy':>10} {'Numba':>10} {'Parallel':>10}")
    print("-" * 66)

    # normalize_rows
    t1 = bench(normalize_rows_python, X)
    t2 = bench(normalize_rows_numpy, X)
    print(f"{'normalize_rows':<22} {t1*1e3:>9.1f}ms {t2*1e3:>9.3f}ms {'—':>10} {'—':>10}")

    # softmax
    t1 = bench(softmax_python, logits)
    t2 = bench(softmax_numpy, logits)
    print(f"{'softmax':<22} {t1*1e3:>9.1f}ms {t2*1e3:>9.3f}ms {'—':>10} {'—':>10}")

    # pairwise_distances
    t1 = bench(pairwise_distances_python, X_sub, Y)
    t2 = bench(pairwise_distances_numpy, X_sub, Y)
    t3 = bench(pairwise_distances_numba, X_sub, Y)
    t4 = bench(pairwise_distances_parallel, X_sub, Y)
    print(f"{'pairwise_dist':<22} {t1*1e3:>9.1f}ms {t2*1e3:>9.3f}ms {t3*1e3:>9.3f}ms {t4*1e3:>9.3f}ms")

    print("-" * 66)
    print("Fill in the table in your notes. Which is fastest? Why?")


if __name__ == "__main__":
    main()