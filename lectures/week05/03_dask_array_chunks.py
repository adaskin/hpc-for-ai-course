"""
Week 5 · Dask Array: lazy, chunked arrays. Chunk size is the granularity knob again.

1. The array is only a recipe; chunks are created, used and dropped during compute().
2. Sweep the chunk size: same work, different number of tasks.

Make SHAPE larger than your RAM for a real out-of-core run: the array is never in
memory all at once.

Run:  python 03_dask_array_chunks.py
"""
import resource

import dask.array as da

from corpus import Timer

SHAPE = (20_000, 5_000)                    # 100 million float64 values = 0.8 GB


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # Linux: KB -> MB


if __name__ == "__main__":
    # ---- 1. lazy and chunked ----
    x = da.ones(SHAPE, chunks=(2_000, 5_000))
    chunk_mb = x.chunksize[0] * x.chunksize[1] * 8 / 1e6
    print(f"array: {x.nbytes / 1e9:.2f} GB in {x.numblocks[0]} x {x.numblocks[1]} chunks "
          f"of {chunk_mb:.0f} MB  (nothing computed yet)")

    expr = (x * 2 + 1).sum()                                   # still lazy
    with Timer() as t:
        total = expr.compute()
    assert total == 3 * SHAPE[0] * SHAPE[1]
    print(f"sum = {total:.0f} in {t.s:.2f} s | peak memory {peak_mb():.0f} MB "
          f"vs array size {x.nbytes / 1e6:.0f} MB")

    # ---- 2. chunk-size sweep ----
    print(f"\n{'rows/chunk':>10} {'chunks':>7} {'tasks':>6} {'chunk MB':>9} {'time s':>8}")
    for rows in (50, 500, 5_000, 20_000):
        z = da.ones(SHAPE, chunks=(rows, 5_000))
        e = (z * 2 + 1).sum()
        n_tasks = len(e.__dask_graph__())
        with Timer() as t:
            r = e.compute()
        assert r == total
        print(f"{rows:>10} {z.numblocks[0]:>7} {n_tasks:>6} {rows * 5_000 * 8 / 1e6:>9.0f} {t.s:>8.2f}")
    print("\nDask docs suggest chunks of about 100 MB or more: too small = task overhead,"
          "\ntoo big = no parallelism and memory pressure.")

    # ---- 3. the graph of a tiny array (needs Graphviz) ----
    try:
        da.ones((4, 4), chunks=2).sum().visualize(filename="array_graph.png")
        print("small array graph written to array_graph.png")
    except Exception as e:
        print("could not draw the graph (is Graphviz installed?):", e)
