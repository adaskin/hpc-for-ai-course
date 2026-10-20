"""
20 - Same sum, two graphs: the order you submit decides the critical path.

Idea and code shape from R. Nishihara's Ray talks (see the Sources slide):
add 8 values two at a time, where every `add` takes 0.5 s.

    chain:  vals = [new_val] + vals[2:]   # the next add needs the result we just launched
    tree:   vals = vals[2:] + [new_val]   # the next add uses two inputs that are ready now

Part A needs nothing but Python: it builds both graphs, measures the critical path
and schedules them on W workers.
Part B / C run the real thing with Dask `delayed` and Ray, if they are installed.

    python 20_tree_reduction.py
"""
import heapq
import math
import time


# ---------------------------------------------------------------- A: pure Python
def build(order, n):
    """Return the list of adds as (left, right) pairs. Inputs are 0..n-1; add k has id n+k."""
    vals, adds = list(range(n)), []
    while len(vals) > 1:
        new = n + len(adds)
        adds.append((vals[0], vals[1]))
        vals = [new] + vals[2:] if order == "chain" else vals[2:] + [new]
    return adds


def depth(adds, n):
    d = {i: 0 for i in range(n)}
    for k, (a, b) in enumerate(adds):
        d[n + k] = 1 + max(d[a], d[b])
    return d[n + len(adds) - 1]


def makespan(adds, n, t, workers):
    """Greedy list scheduling in submission order: start = max(inputs ready, a worker is free)."""
    finish = {i: 0.0 for i in range(n)}
    free = [0.0] * workers
    heapq.heapify(free)
    for k, (a, b) in enumerate(adds):
        start = max(finish[a], finish[b], heapq.heappop(free))
        finish[n + k] = start + t
        heapq.heappush(free, finish[n + k])
    return finish[n + len(adds) - 1]


def part_a():
    t = 0.5
    print("A) critical path and schedule  (each add takes 0.5 s)\n")
    print(f"{'values':>7} {'order':>6} {'adds':>5} {'depth':>6} {'1 worker':>9} {'2 workers':>10} {'4 workers':>10} {'enough':>8}")
    for n in (8, 64):
        for order in ("chain", "tree"):
            adds = build(order, n)
            d = depth(adds, n)
            row = [makespan(adds, n, t, w) for w in (1, 2, 4, n)]
            print(f"{n:>7} {order:>6} {len(adds):>5} {d:>6} " + " ".join(f"{x:>9.1f}s" for x in row))
            assert d == (n - 1 if order == "chain" else math.ceil(math.log2(n)))
    print("\n   chain: depth n-1, extra workers never help.  tree: depth log2(n), width n/2.\n")


# ---------------------------------------------------------------- B: Dask delayed
def add_slow(x, y):
    time.sleep(0.5)
    return x + y


def run_graph(order, submit, n=8):
    vals = list(range(1, n + 1))
    while len(vals) > 1:
        new = submit(vals[0], vals[1])
        vals = [new] + vals[2:] if order == "chain" else vals[2:] + [new]
    return vals[0]


def part_b():
    try:
        import dask
        from dask import delayed
    except ImportError:
        print("B) Dask not installed: skipped\n")
        return
    print("B) Dask delayed, scheduler='threads' (sleep releases the GIL, so 4 threads can overlap)\n")
    dly = delayed(add_slow)
    times = {}
    for order in ("chain", "tree"):
        total = run_graph(order, dly)
        t0 = time.perf_counter()
        result = total.compute(scheduler="threads", num_workers=4)
        times[order] = time.perf_counter() - t0
        assert result == 36
        print(f"   {order:>5}: {times[order]:.2f} s   (graph built first, nothing ran until compute)")
    assert times["tree"] < times["chain"]
    print()


# ---------------------------------------------------------------- C: Ray
def part_c():
    try:
        import ray
    except ImportError:
        print("C) Ray not installed: skipped\n")
        return
    print("C) Ray tasks, 4 declared CPUs (the tasks only sleep, so 2 real cores are enough)\n")
    ray.init(num_cpus=4, include_dashboard=False, logging_level="ERROR")
    add = ray.remote(add_slow)
    times = {}
    try:
        for order in ("chain", "tree"):
            t0 = time.perf_counter()
            result = ray.get(run_graph(order, add.remote))      # only ONE ray.get, at the end
            times[order] = time.perf_counter() - t0
            assert result == 36
            print(f"   {order:>5}: {times[order]:.2f} s")
    finally:
        ray.shutdown()
    assert times["tree"] < times["chain"]
    print()


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
