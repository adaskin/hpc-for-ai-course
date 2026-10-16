"""
Shared helpers for the Week 5 examples (driver side only).

The functions that run on Dask / Ray workers are defined inside each example
script, so the workers receive them by value and need no access to this file.
"""
import math
import random
import time
from collections import Counter


def make_corpus(n_docs=2000, words_per_doc=2000, vocab=5000, seed=0):
    """A synthetic corpus: n_docs documents of random words w0 .. w{vocab-1}."""
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    return [" ".join(rng.choices(words, k=words_per_doc)) for _ in range(n_docs)]


def chunk(seq, size):
    """Split a list into consecutive pieces of `size` items (the last may be shorter)."""
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def df_reference(corpus):
    """Serial document frequency: in how many documents does each word occur?"""
    df = Counter()
    for doc in corpus:
        df.update(set(doc.split()))
    return df


def idf_from_df(df, n_docs):
    return {w: math.log(n_docs / c) for w, c in df.items()}


def weight_reference(corpus, idf):
    """Serial reference for the 'weight' step: sum of idf over every token."""
    return sum(idf[w] for doc in corpus for w in doc.split())


class Timer:
    """with Timer() as t: ...   then t.s is the elapsed wall-clock time in seconds."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0
