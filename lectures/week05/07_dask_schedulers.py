"""
Week 5 · Which Dask scheduler? The GIL from Week 4 decides.

Same graph, three schedulers:
  synchronous   one thread; the debugging baseline
  threads       default for Array / DataFrame / delayed; parallel only where the GIL is released
  processes     default for Bag; real parallelism for pure Python, but arguments are pickled

Expect: NumPy tasks speed up with threads; pure-Python tasks need processes.

Run:  python 07_dask_schedulers.py
"""
import random

import numpy as np
import dask
from dask import delayed

from corpus import Timer


def numpy_task(seed):
    """NumPy's random generator and sort release the GIL."""
    rng = np.random.default_rng(seed)
    return float(np.sort(rng.random(3_000_000))[1_500_000])


def python_task(seed):
    """A pure-Python loop holds the GIL."""
    rnd = random.Random(seed)
    counts = {}
    for _ in range(400_000):
        w = rnd.randrange(1_000)
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values())


def compare(label, fn):
    tasks = [delayed(fn)(i) for i in range(8)]
    print(f"\n{label}")
    reference = None
    for sched in ("synchronous", "threads", "processes"):
        with Timer() as t:
            out = dask.compute(*tasks, scheduler=sched)
        if reference is None:
            reference = out
        assert out == reference                      # same answers whatever the scheduler
        print(f"  {sched:<12} {t.s:6.2f} s")


if __name__ == "__main__":
    compare("NumPy-heavy tasks (GIL released)", numpy_task)
    compare("pure-Python tasks (GIL held)", python_task)
    print("\nprocesses pays a start-up and pickling cost; with only 2 cores the best case is ~2x.")
