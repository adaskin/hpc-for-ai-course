"""
Week 5 – Granularity sweep.

How much work should one task carry? Splits a corpus into chunks of
`docs_per_task` documents, counts words in parallel (map), then merges the
partial counts serially (reduce). Also times the plain serial version, so the speedup
S = T_serial / T_parallel can be compared with Amdahl's ceiling (S <= number of workers).
Prints a table and writes granularity_results.csv.

Note: process-pool start-up is included in the map time on purpose –
it is part of the real cost of going parallel.
Run:  python 01_granularity_sweep.py
"""
import csv
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

N_DOCS, WORDS_PER_DOC, VOCAB, WORKERS, REPEATS = 2000, 2000, 5000, 2, 5

random.seed(0)
_vocab = [f"w{i}" for i in range(VOCAB)]
CORPUS = [" ".join(random.choices(_vocab, k=WORDS_PER_DOC)) for _ in range(N_DOCS)]


def count_chunk(docs):
    """Pure-Python word count for one chunk of documents."""
    c = Counter()
    for d in docs:
        c.update(d.split())
    return c


def run_once(docs_per_task):
    chunks = [CORPUS[i:i + docs_per_task] for i in range(0, N_DOCS, docs_per_task)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as ex:
        partials = list(ex.map(count_chunk, chunks))
    map_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    total = Counter()
    for p in partials:          # the serial reduce
        total.update(p)
    reduce_t = time.perf_counter() - t0
    return len(chunks), map_t, reduce_t


def serial_time():
    """The same work in one process: no pool, no pickling, no merge."""
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        count_chunk(CORPUS)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


if __name__ == "__main__":
    rows = []
    t_serial = serial_time()
    print(f"{N_DOCS} docs x {WORDS_PER_DOC} words, {WORKERS} workers, median of {REPEATS}")
    print(f"serial (1 process): {t_serial:.3f} s   Amdahl ceiling with {WORKERS} workers: {WORKERS}x\n")
    print(f"{'docs/task':>9} {'tasks':>6} {'map s':>8} {'reduce s':>9} {'total s':>8} {'speedup':>8}")
    for dpt in (1, 10, 100, 1000):
        runs = [run_once(dpt) for _ in range(REPEATS)]
        n = runs[0][0]
        m = statistics.median(r[1] for r in runs)
        r = statistics.median(r[2] for r in runs)
        rows.append((dpt, n, m, r))
        print(f"{dpt:>9} {n:>6} {m:>8.3f} {r:>9.4f} {m + r:>8.3f} {t_serial / (m + r):>7.2f}x")
    with open("granularity_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["docs_per_task", "tasks", "map_s", "reduce_s", "serial_s", "workers"])
        w.writerows([(*row, t_serial, WORKERS) for row in rows])
