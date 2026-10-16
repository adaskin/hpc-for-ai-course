"""
Week 5 · The whole DF -> IDF pipeline in Ray. Compare with 02_dask_delayed_graph.py.

Same graph (count -> tree merge -> IDF -> weight), but here the graph is DYNAMIC: it is
built by ordinary Python loops while the program runs. Chunks are put in the object store
once and used twice (count step and weight step).

Needs:  pip install ray
Run:    python 16_ray_df_idf.py
"""
import logging
import math
from collections import Counter

import ray

from corpus import Timer, chunk, df_reference, idf_from_df, make_corpus, weight_reference


@ray.remote
def df_chunk(docs):
    c = Counter()
    for d in docs:
        c.update(set(d.split()))
    return c


@ray.remote
def merge(a, b):
    return a + b


@ray.remote
def make_idf(df, n_docs):
    return {w: math.log(n_docs / c) for w, c in df.items()}


@ray.remote
def weight_chunk(docs, idf):
    return sum(idf[w] for d in docs for w in d.split())


def tree_reduce(refs):
    """Pairwise merge of futures until one is left."""
    while len(refs) > 1:
        nxt = [merge.remote(refs[i], refs[i + 1]) for i in range(0, len(refs) - 1, 2)]
        if len(refs) % 2:
            nxt.append(refs[-1])
        refs = nxt
    return refs[0]


if __name__ == "__main__":
    ray.init(num_cpus=2, include_dashboard=False, logging_level=logging.ERROR)
    corpus = make_corpus(n_docs=2000, words_per_doc=500)
    chunk_refs = [ray.put(c) for c in chunk(corpus, 100)]             # 20 chunks, stored once

    with Timer() as t_serial:
        ref_idf = idf_from_df(df_reference(corpus), len(corpus))
        ref = weight_reference(corpus, ref_idf)

    with Timer() as t_ray:
        counts = [df_chunk.remote(c) for c in chunk_refs]             # width 20
        total_df = tree_reduce(counts)                                # width 10, 5, 3, 2, 1
        idf_ref = make_idf.remote(total_df, len(corpus))              # the narrow waist
        weights = [weight_chunk.remote(c, idf_ref) for c in chunk_refs]   # width 20 again
        result = sum(ray.get(weights))

    assert math.isclose(result, ref, rel_tol=1e-9), (result, ref)
    print(f"serial {t_serial.s:.2f} s | Ray {t_ray.s:.2f} s | result matches the serial reference")
    ray.shutdown()
