"""
Week 5 · Dask Bag: map / flatten / frequencies on unstructured text (document frequency again).

A Bag is a collection of Python objects split into partitions. It runs on the
*multiprocessing* scheduler by default, which suits pure-Python work (the GIL, Week 4).
The sweep shows granularity once more: too many partitions = overhead.

Run:  python 05_dask_bag_df_idf.py
"""
import dask.bag as db

from corpus import Timer, df_reference, make_corpus


def df_bag(corpus, npartitions):
    bag = db.from_sequence(corpus, npartitions=npartitions)
    return bag.map(lambda d: set(d.split())).flatten().frequencies()


if __name__ == "__main__":
    corpus = make_corpus(n_docs=2000, words_per_doc=1000)

    with Timer() as t_ser:
        ref = df_reference(corpus)
    print(f"serial Counter: {t_ser.s:.2f} s")

    # warm-up: the first call pays one-off start-up costs
    dict(df_bag(corpus, 2).compute(scheduler="processes"))

    print(f"\n{'partitions':>10} {'time s':>8}")
    for n in (2, 8, 50, 500):
        with Timer() as t:
            result = dict(df_bag(corpus, n).compute(scheduler="processes"))
        assert result == dict(ref)                       # same document frequencies as the serial code
        print(f"{n:>10} {t.s:>8.2f}")

    print("\nfirst three (word, document frequency) pairs:", list(result.items())[:3])
