"""
Week 5 · Granularity with Ray: the same sweep as 01_granularity_sweep.py, but with Ray tasks.

Tiny tasks pay Ray's per-task cost (scheduling + moving a result back through the object
store) thousands of times. The first row is slow on purpose.
If granularity_results.csv from example 01 is present, its totals are shown for comparison.

Needs:  pip install ray
Run:    python 13_ray_granularity.py
"""
import csv
import logging
from collections import Counter
from pathlib import Path

import ray

from corpus import Timer, chunk, make_corpus

N_DOCS, WORDS_PER_DOC = 2000, 2000


@ray.remote
def count_chunk(docs):
    c = Counter()
    for d in docs:
        c.update(d.split())
    return c


def run_once(corpus, docs_per_task):
    chunks = chunk(corpus, docs_per_task)
    with Timer() as t_map:
        parts = ray.get([count_chunk.remote(c) for c in chunks])
    with Timer() as t_red:
        total = Counter()
        for p in parts:
            total.update(p)
    return len(chunks), t_map.s, t_red.s, total


if __name__ == "__main__":
    ray.init(num_cpus=2, include_dashboard=False, logging_level=logging.ERROR)
    corpus = make_corpus(N_DOCS, WORDS_PER_DOC)
    ref = Counter()
    for d in corpus:
        ref.update(d.split())
    ray.get(count_chunk.remote(corpus[:10]))                 # warm-up

    csv_path = Path(__file__).with_name("granularity_results.csv")
    pool_total = {}
    if csv_path.exists():
        with open(csv_path) as f:
            pool_total = {int(r["docs_per_task"]): float(r["map_s"]) + float(r["reduce_s"])
                          for r in csv.DictReader(f)}

    print(f"{N_DOCS} docs x {WORDS_PER_DOC} words, 2 CPUs")
    print(f"{'docs/task':>9} {'tasks':>6} {'map s':>8} {'reduce s':>9} {'total s':>8} {'pool total (01)':>16}")
    for dpt in (1, 10, 100, 1000):
        n, m, r, total = run_once(corpus, dpt)
        assert total == ref
        other = f"{pool_total[dpt]:.2f}" if dpt in pool_total else "-"
        print(f"{dpt:>9} {n:>6} {m:>8.2f} {r:>9.3f} {m + r:>8.2f} {other:>16}")

    ray.shutdown()
