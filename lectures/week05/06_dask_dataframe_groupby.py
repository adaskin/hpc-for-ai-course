"""
Week 5 · Dask DataFrame: partitions, reductions, and what a shuffle costs.

  column mean     a REDUCE: each partition sends one small result
  groupby mean    partial result per group in every partition, merged in a tree
  shuffle         explicit repartition by key: rows move between partitions
  set_index       sorts by a key (a shuffle) ONCE; afterwards .loc touches few partitions

Run:  python 06_dask_dataframe_groupby.py
"""
import numpy as np
import pandas as pd
import dask.dataframe as dd

from corpus import Timer

N, AUTHORS = 2_000_000, 50_000

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pdf = pd.DataFrame({"author": rng.integers(0, AUTHORS, N),
                        "length": rng.integers(10, 5_000, N)})
    ddf = dd.from_pandas(pdf, npartitions=8)
    print(f"{len(pdf):,} rows in {ddf.npartitions} partitions\n")

    with Timer() as t_mean:
        mean_len = ddf["length"].mean().compute()
    assert np.isclose(mean_len, pdf["length"].mean())

    with Timer() as t_grp:
        per_author = ddf.groupby("author")["length"].mean().compute()
    ref = pdf.groupby("author")["length"].mean()
    pd.testing.assert_series_equal(per_author.sort_index(), ref.sort_index(), check_names=False)

    with Timer() as t_shuf:
        sizes = ddf.shuffle(on="author").map_partitions(len, meta=("rows", "int64")).compute()
    assert int(sizes.sum()) == N                         # no row lost or duplicated

    print(f"mean of one column            {t_mean.s:7.3f} s   (reduce)")
    print(f"groupby mean, {AUTHORS:,} groups  {t_grp.s:7.3f} s   (partial results per group)")
    print(f"shuffle by author             {t_shuf.s:7.3f} s   (rows move between partitions)")

    # ---- pay for the sort once, then lookups are cheap ----
    with Timer() as t_idx:
        indexed = ddf.set_index("author").persist()
    with Timer() as t_loc:
        hits = indexed.loc[100:200].compute()
    expected = int(((pdf["author"] >= 100) & (pdf["author"] <= 200)).sum())
    assert len(hits) == expected
    print(f"\nset_index('author')           {t_idx.s:7.3f} s   (known divisions: {indexed.known_divisions})")
    print(f".loc[100:200] afterwards      {t_loc.s:7.3f} s   ({len(hits):,} rows)")
