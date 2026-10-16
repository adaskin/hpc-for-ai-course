"""
Week 5 · Dask `delayed`: build the DF -> IDF task graph, look at it, then run it.

Nothing runs until compute(). Open df_idf_graph.png and compare it with the
graph on the slide (4 counts -> 2 merges -> merge all -> IDF -> 4 weights).
The picture needs the Graphviz *binary* (Colab: !apt-get install graphviz).

Run:  python 02_dask_delayed_graph.py
"""
import math
from collections import Counter

import dask
from dask import delayed

from corpus import Timer, chunk, df_reference, idf_from_df, make_corpus, weight_reference


def df_chunk(docs):
    """Document frequency of one chunk."""
    c = Counter()
    for d in docs:
        c.update(set(d.split()))
    return c


def merge(a, b):
    return a + b                                   # Counter + Counter adds the counts


def make_idf(df, n_docs):
    return {w: math.log(n_docs / c) for w, c in df.items()}


def weight_chunk(docs, idf):
    return sum(idf[w] for d in docs for w in d.split())


if __name__ == "__main__":
    corpus = make_corpus(n_docs=2000, words_per_doc=500)
    chunks = chunk(corpus, len(corpus) // 4)                    # 4 chunks

    # ---- build the graph: no work happens in this block ----
    counts = [delayed(df_chunk)(c) for c in chunks]             # width 4
    m1 = delayed(merge)(counts[0], counts[1])                   # width 2
    m2 = delayed(merge)(counts[2], counts[3])
    total = delayed(merge)(m1, m2)                              # the narrow waist
    idf = delayed(make_idf)(total, len(corpus))
    weights = [delayed(weight_chunk)(c, idf) for c in chunks]   # width 4 again

    try:
        dask.visualize(*weights, filename="df_idf_graph", format="png")
        print("graph written to df_idf_graph.png")
    except Exception as e:                                      # Graphviz binary missing
        print("could not draw the graph (is Graphviz installed?):", e)

    # ---- run it ----
    with Timer() as t:
        parts = dask.compute(*weights)               # delayed uses the threaded scheduler by default
    print(f"compute: {t.s:.2f} s  (pure-Python tasks on threads: the GIL limits the speedup)")

    # ---- check against the plain serial version ----
    ref_idf = idf_from_df(df_reference(corpus), len(corpus))
    ref = weight_reference(corpus, ref_idf)
    assert math.isclose(sum(parts), ref, rel_tol=1e-9), (sum(parts), ref)
    print("result matches the serial reference")
