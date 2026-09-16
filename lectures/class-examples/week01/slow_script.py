"""
YZM 513 – Week 1: cProfile demo.

A K-means-style iteration written with Python loops (slow on purpose).

Run:
    python slow_script.py                             # plain run
    python -m cProfile -s cumulative slow_script.py   # profiled run

Expect compute_distances to dominate, assign_clusters second,
and the simulated I/O (time.sleep) visible but small.
"""

import time

import numpy as np

N_POINTS = 200
N_DIMS = 10
N_CENTERS = 5
N_ITERS = 200


def compute_distances(points, centers):
    """Pairwise distances via nested Python loops.  <- the hot spot"""
    n, _ = points.shape
    k = centers.shape[0]
    dist = np.empty((n, k))
    for i in range(n):
        for j in range(k):
            diff = points[i] - centers[j]
            dist[i, j] = np.sqrt(np.sum(diff ** 2))
    return dist


def assign_clusters(dist):
    """Nearest-centre argmin via Python loops."""
    n, k = dist.shape
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        best = 0
        for j in range(1, k):
            if dist[i, j] < dist[i, best]:
                best = j
        labels[i] = best
    return labels


def update_centers(points, labels, k):
    """Mean of assigned points per cluster (already vectorised)."""
    new_centers = np.zeros((k, points.shape[1]))
    for j in range(k):
        mask = labels == j
        if mask.any():
            new_centers[j] = points[mask].mean(axis=0)
    return new_centers


def main():
    rng = np.random.default_rng(42)
    points = rng.random((N_POINTS, N_DIMS))
    centers = rng.random((N_CENTERS, N_DIMS))

    for it in range(N_ITERS):
        dist = compute_distances(points, centers)
        labels = assign_clusters(dist)
        centers = update_centers(points, labels, N_CENTERS)

        if it % 40 == 0:
            time.sleep(0.01)  # simulate I/O, e.g. checkpointing

    inertia = dist[np.arange(N_POINTS), labels].sum()
    print(f"done: {N_ITERS} iterations, final inertia = {inertia:.3f}")


if __name__ == "__main__":
    main()