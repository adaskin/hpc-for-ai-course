"""
Week 2 – Benchmark: Python loops vs NumPy vectorised versions.
Run: python benchmark_numpy.py
"""
import time
import numpy as np


# ── Original slow versions (from Week 1) ──────────────────────────

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


# ── NumPy vectorised versions ─────────────────────────────────────

def normalize_rows_numpy(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def softmax_numpy(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def pairwise_distances_numpy(A, B):
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)
    B_sq = np.sum(B ** 2, axis=1, keepdims=True)
    cross = A @ B.T
    dist_sq = A_sq + B_sq.T - 2 * cross
    return np.sqrt(np.maximum(dist_sq, 0))


# ── Benchmark ─────────────────────────────────────────────────────

def bench(fn, *args, n_runs=5):
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

    print("=" * 60)
    print(f"{'Function':<25} {'Python (ms)':>12} {'NumPy (ms)':>12} {'Speedup':>10}")
    print("=" * 60)

    # normalize_rows
    t_py = bench(normalize_rows_python, X) * 1000
    t_np = bench(normalize_rows_numpy, X) * 1000
    print(f"{'normalize_rows':<25} {t_py:>12.2f} {t_np:>12.4f} {t_py/t_np:>9.0f}×")

    # softmax
    t_py = bench(softmax_python, logits) * 1000
    t_np = bench(softmax_numpy, logits) * 1000
    print(f"{'softmax':<25} {t_py:>12.2f} {t_np:>12.4f} {t_py/t_np:>9.0f}×")

    # pairwise_distances (use subset to keep Python version manageable)
    X_sub = X[:200]
    t_py = bench(pairwise_distances_python, X_sub, Y) * 1000
    t_np = bench(pairwise_distances_numpy, X_sub, Y) * 1000
    print(f"{'pairwise_distances':<25} {t_py:>12.2f} {t_np:>12.4f} {t_py/t_np:>9.0f}×")

    print("=" * 60)


if __name__ == "__main__":
    main()