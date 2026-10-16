"""
Week 5 – Granularity sweep.

How much work should one task carry? Splits a corpus into chunks of
`docs_per_task` documents, counts words in parallel (map), then merges the
partial counts serially (reduce). Prints a table and writes granularity_results.csv.

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

N_DOCS, WORDS_PER_DOC, VOCAB, WORKERS, REPEATS = 2000, 2000, 5000, 2, 3

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


if __name__ == "__main__":
    rows = []
    print(f"{N_DOCS} docs x {WORDS_PER_DOC} words, {WORKERS} workers, median of {REPEATS}")
    print(f"{'docs/task':>9} {'tasks':>6} {'map s':>8} {'reduce s':>9} {'total s':>8}")
    for dpt in (1, 10, 100, 1000):
        runs = [run_once(dpt) for _ in range(REPEATS)]
        n = runs[0][0]
        m = statistics.median(r[1] for r in runs)
        r = statistics.median(r[2] for r in runs)
        rows.append((dpt, n, m, r))
        print(f"{dpt:>9} {n:>6} {m:>8.3f} {r:>9.4f} {m + r:>8.3f}")
    with open("granularity_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["docs_per_task", "tasks", "map_s", "reduce_s"])
        w.writerows(rows)
