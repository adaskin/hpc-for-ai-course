"""
Week 5 · Dask tutorial, one step at a time.

    python 17_dask_tutorial.py 1        # run one step (1-9)
    python 17_dask_tutorial.py all      # run every step in order

    1  delayed: build a lazy graph, draw it, compute it
    2  schedulers: synchronous / threads / processes (dask.config, num_workers)
    3  collections: Array, Bag, DataFrame (each checked against plain NumPy/pandas)
    4  LocalCluster + Client: start, inspect, use, shut down
    5  sizing the cluster: processes vs threads (the GIL)
    6  futures: submit, map, as_completed, futures as arguments
    7  data: scatter(broadcast=True) and persist
    8  watch it run: dashboard link and performance_report
    9  manage it: scale, restart, cancel, close

Needs:  pip install "dask[complete]"      (Colab, step 1 picture: apt-get install graphviz)
Keep every cluster inside  if __name__ == "__main__":  (workers re-import this file).
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")      # BLAS threads per worker = 1; child processes inherit it
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import dask
import dask.array as da
import dask.bag as db
from dask import delayed
from dask.distributed import Client, LocalCluster, as_completed, performance_report


def burn(n):
    """Pure-Python CPU work: holds the GIL."""
    return sum(i * i for i in range(n))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------- 1
def step1_delayed():
    @delayed
    def inc(x):
        return x + 1

    @delayed
    def add(a, b):
        return a + b

    total = add(inc(1), inc(2))              # builds a graph, runs nothing
    print("lazy object :", total)
    try:
        dask.visualize(total, filename="tutorial_graph", format="png")
        print("graph drawn : tutorial_graph.png")
    except Exception as e:                    # graphviz missing
        print("(no picture:", type(e).__name__, "- install graphviz)")
    print("computed    :", total.compute())  # 5
    assert total.compute() == 5


# ---------------------------------------------------------------- 2
def step2_schedulers():
    tasks = [delayed(burn)(1_500_000) for _ in range(8)]
    results = {}
    for sched in ("synchronous", "threads", "processes"):
        out, t = timed(dask.compute, *tasks, scheduler=sched, num_workers=2)
        results[sched] = out
        print(f"scheduler={sched:<12} {t:5.2f} s")
    assert results["synchronous"] == results["threads"] == results["processes"]
    # the same choice, made once for a whole block of code:
    with dask.config.set(scheduler="processes", num_workers=2):
        _, t = timed(dask.compute, *tasks)
    print(f"dask.config.set(scheduler='processes', num_workers=2): {t:.2f} s")
    print("pure Python holds the GIL: expect threads ~ synchronous, processes faster")


# ---------------------------------------------------------------- 3
def step3_collections():
    # Array: chunked NumPy
    x = da.random.random((4000, 4000), chunks=(1000, 1000))
    print("Array  : blocks", x.numblocks, "sum =", round(float(x.sum().compute()), 1))
    # Bag: a list of Python objects
    squares = db.from_sequence(range(10), npartitions=2).map(lambda v: v * v)
    print("Bag    : sum of squares =", squares.sum().compute())
    assert squares.sum().compute() == sum(v * v for v in range(10))
    # DataFrame: chunked pandas
    import pandas as pd
    import dask.dataframe as dd
    pdf = pd.DataFrame({"author": list("abcab") * 100, "n": range(500)})
    ddf = dd.from_pandas(pdf, npartitions=4)
    got = ddf.groupby("author")["n"].mean().compute().sort_index()
    want = pdf.groupby("author")["n"].mean().sort_index()
    assert np.allclose(got.values, want.values)
    print("DataFrame groupby mean matches pandas:", dict(got.round(1)))


# ---------------------------------------------------------------- 4
def step4_cluster():
    with LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="1GB") as cluster, \
         Client(cluster) as client:
        print(client)                                    # <Client: ... workers=2, threads=2, memory=...>
        print("dashboard   :", client.dashboard_link)
        print("threads     :", client.nthreads())        # {worker address: number of threads}
        print("1 + 2 + 3   :", client.submit(sum, [1, 2, 3]).result())
    print("cluster closed (the with-block shut the workers down)")


# ---------------------------------------------------------------- 5
def _run(cfg, n_tasks=8, n=1_500_000):
    with LocalCluster(**cfg) as cluster, Client(cluster) as client:
        client.submit(burn, 10).result()                 # warm-up: workers are up
        t0 = time.perf_counter()
        client.gather(client.map(burn, [n] * n_tasks))
        return time.perf_counter() - t0


def step5_sizing():
    a = _run(dict(n_workers=2, threads_per_worker=1))
    b = _run(dict(n_workers=1, threads_per_worker=2))
    print(f"2 workers x 1 thread : {a:5.2f} s   <- two processes, two GILs")
    print(f"1 worker  x 2 threads: {b:5.2f} s   <- one process, one GIL")
    print("pure Python: many single-threaded workers. NumPy/pandas (GIL released): fewer workers, more threads.")


# ---------------------------------------------------------------- 6
def step6_futures():
    with LocalCluster(n_workers=2, threads_per_worker=1) as cluster, Client(cluster) as client:
        futs = client.map(burn, [200_000 * k for k in range(1, 7)])   # starts running at once
        for f in as_completed(futs):                                  # finish order, not submit order
            print("done:", f.key[:12], f.result())
        total = client.submit(sum, futs)                              # futures as arguments
        print("sum of all :", total.result())
        assert total.result() == sum(burn(200_000 * k) for k in range(1, 7))


# ---------------------------------------------------------------- 7
def _row_sum(m, i):
    return float(m[i].sum())


def step7_data():
    with LocalCluster(n_workers=2, threads_per_worker=1) as cluster, Client(cluster) as client:
        big = np.random.default_rng(0).random((1500, 1500))           # 18 MB
        big_f = client.scatter(big, broadcast=True)                   # ship once to every worker
        futs = [client.submit(_row_sum, big_f, i) for i in range(8)]  # tasks carry a reference
        got = client.gather(futs)
        assert np.allclose(got, big[:8].sum(axis=1))
        print("scatter: 8 tasks used one 18 MB array, shipped once per worker")
        x = da.random.random((6000, 6000), chunks=(1500, 1500)).persist()   # computed, kept on the workers
        print("persist: first sum", round(float(x.sum().compute()), 1),
              "| second sum reuses the stored chunks:", round(float(x.sum().compute()), 1))


# ---------------------------------------------------------------- 8
def step8_watch():
    with LocalCluster(n_workers=2, threads_per_worker=1) as cluster, Client(cluster) as client:
        print("open the dashboard now:", client.dashboard_link)
        try:
            with performance_report(filename="dask-report.html"):
                client.gather(client.map(burn, [800_000] * 12))
            print("wrote dask-report.html  (open it in a browser: task stream, worker profile)")
        except ImportError as e:
            print("performance_report needs bokeh:", e)


# ---------------------------------------------------------------- 9
def step9_manage():
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    try:
        cluster.scale(3)                                  # ask for 3 workers
        client.wait_for_workers(3)
        print("workers after scale(3):", len(client.scheduler_info()["workers"]))
        slow = client.submit(time.sleep, 60)
        time.sleep(0.5)
        slow.cancel()                                     # give up on one task
        print("cancelled task status :", slow.status)
        client.restart()                                  # restart every worker, forget all futures
        print("workers after restart :", len(client.scheduler_info()["workers"]))
    finally:
        client.close()                                    # always close the client, then the cluster
        cluster.close()
    print("closed")


STEPS = {1: step1_delayed, 2: step2_schedulers, 3: step3_collections, 4: step4_cluster,
         5: step5_sizing, 6: step6_futures, 7: step7_data, 8: step8_watch, 9: step9_manage}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = list(STEPS) if arg == "all" else [int(arg)]
    for k in todo:
        print(f"\n=== step {k}: {STEPS[k].__name__} ===")
        STEPS[k]()
