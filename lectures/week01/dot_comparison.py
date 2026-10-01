"""
YZM 513 – Week 1: Python loop vs. NumPy (BLAS) dot product.

Run:
    python dot_comparison.py

Expected on a typical laptop (N = 10^6):
    Python loop:  ~0.2 s
    NumPy dot:    ~0.0005 s
    Speedup:      ~400x
"""

import time

import numpy as np

N = 1_000_000


def dot_python(a, b):
    """Dot product element-by-element, through the interpreter."""
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def dot_numpy(a, b):
    """One call into compiled BLAS."""
    return np.dot(a, b)


def bench(fn, *args, repeat=5):
    """Best-of-`repeat` wall-clock time in seconds."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    rng = np.random.default_rng(0)
    a = rng.random(N)
    b = rng.random(N)

    # Sanity check on a small slice before timing the slow loop.
    assert np.isclose(dot_python(a[:1000], b[:1000]),
                      np.dot(a[:1000], b[:1000]))

    t_py = bench(dot_python, a, b, repeat=3)  # slow: 3 runs are enough
    t_np = bench(dot_numpy, a, b)

    print(f"Python loop:  {t_py:.4f} s")
    print(f"NumPy dot:    {t_np:.6f} s")
    print(f"Speedup:      {t_py / t_np:.0f}×")


if __name__ == "__main__":
    main()