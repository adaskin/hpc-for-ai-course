"""
YZM 513 – Week 1 exercise: three deliberately slow functions.

Steps:
1. Run and note total time:
       python exercise_week1.py
2. Function-level profile:
       python -m cProfile -s cumulative exercise_week1.py
3. Line-level profile of pairwise_distances:
       - uncomment the @profile line above it
       - kernprof -l -v exercise_week1.py

Write down: top 3 bottlenecks? WHY is each slow? What's the fix?
(Don't implement yet — that's Week 2.)
"""

import numpy as np

# `kernprof -l` injects `profile` as a builtin; fall back to the package
# import; fall back to a no-op so the script also runs with plain python.
try:
    profile  # noqa: F821  (provided by kernprof)
except NameError:
    try:
        from line_profiler import profile
    except ImportError:
        def profile(fn):
            return fn


def normalize_rows(matrix):
    """Row-wise L2 normalisation, pure-Python loops (slow on purpose)."""
    n_rows, n_cols = matrix.shape
    out = np.empty_like(matrix)
    for i in range(n_rows):
        sq_sum = 0.0
        for j in range(n_cols):
            sq_sum += matrix[i, j] ** 2
        norm = np.sqrt(sq_sum)
        for j in range(n_cols):
            out[i, j] = matrix[i, j] / norm
    return out


# @profile  <-- uncomment for the line_profiler task
def pairwise_distances(A, B):
    """Euclidean distances between all rows of A and all rows of B."""
    n, d = A.shape
    m = B.shape[0]
    dist = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                s += diff * diff
            dist[i, j] = np.sqrt(s)
    return dist


def softmax(logits):
    """Row-wise softmax, pure-Python loops (slow on purpose)."""
    n_rows, n_cols = logits.shape
    out = np.empty_like(logits)
    for i in range(n_rows):
        row_max = logits[i, 0]
        for j in range(1, n_cols):
            if logits[i, j] > row_max:
                row_max = logits[i, j]
        total = 0.0
        for j in range(n_cols):
            out[i, j] = np.exp(logits[i, j] - row_max)
            total += out[i, j]
        for j in range(n_cols):
            out[i, j] /= total
    return out


def main():
    rng = np.random.default_rng(0)
    A = rng.random((150, 128))
    B = rng.random((60, 128))
    M = rng.random((800, 256)) * 10.0
    L = rng.normal(scale=5.0, size=(400, 200))

    d = pairwise_distances(A, B)
    m = normalize_rows(M)
    p = softmax(L)

    print("pairwise_distances :", d.shape, f"mean = {d.mean():.4f}")
    print("normalize_rows     :", m.shape,
          f"row norms = {np.linalg.norm(m, axis=1)[:3].round(6)}")
    print("softmax            :", p.shape,
          f"row sums  = {p.sum(axis=1)[:3].round(6)}")


if __name__ == "__main__":
    main()