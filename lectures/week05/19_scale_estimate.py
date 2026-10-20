"""
19 - Project 1 at real scale: a back-of-envelope you can check.

Part VIII slides ("Why It Matters") use these numbers. Nothing here needs Dask or Ray.

What it does
  1. MEASURES this machine's effective scan bandwidth with a matrix-vector product
     (the core of brute-force similarity search is memory-bound, not compute-bound).
  2. Checks that "scatter the query, gather top-k, merge" gives the SAME answer as a
     global top-k  (correctness of the sharded search).
  3. Prints the scale table, the data-moved-per-query table, the indexing-time table
     (Amdahl with a serial fraction) and the failure-probability table.

Everything marked ASSUMED is an input, not a measurement. Change it with the flags and
see how the conclusions move:

    python 19_scale_estimate.py
    python 19_scale_estimate.py --gbps 20     # the numbers used on the slides
    python 19_scale_estimate.py --docs-per-sec 250 --serial-fraction 0.05 --mtbf-days 200
"""
import argparse
import math
import time

import numpy as np

DIM = 768                 # embedding size (a typical sentence-encoder size)
FP32 = 4                  # bytes per float32
ID_BYTES, SCORE_BYTES = 8, 4


def measure_scan_gbps(rows=200_000, dim=DIM, repeats=5):
    """Effective GB/s of a (rows x dim) float32 matrix times a vector, best of `repeats`."""
    X = np.random.default_rng(0).standard_normal((rows, dim), dtype=np.float32)
    q = np.random.default_rng(1).standard_normal(dim, dtype=np.float32)
    X @ q                                              # warm-up
    best = min(_time(lambda: X @ q) for _ in range(repeats))
    return X.nbytes / best / 1e9


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def topk(scores, k):
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(-scores[idx])]


def sharded_topk(X, q, k, n_shards):
    """Each shard scans only its rows and returns k (id, score) pairs; the driver merges."""
    cuts = np.linspace(0, len(X), n_shards + 1, dtype=int)
    cand_ids, cand_scores = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):              # <- on a cluster: one task per shard
        s = X[a:b] @ q
        local = topk(s, min(k, b - a))
        cand_ids.append(local + a)
        cand_scores.append(s[local])
    ids, scores = np.concatenate(cand_ids), np.concatenate(cand_scores)
    best = topk(scores, k)                             # <- the small serial merge
    return ids[best], scores[best]


def check_sharded_search():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50_000, 64), dtype=np.float32)
    q = rng.standard_normal(64, dtype=np.float32)
    want = topk(X @ q, 10)
    for shards in (1, 4, 7, 32):
        got, _ = sharded_topk(X, q, 10, shards)
        assert list(got) == list(want), f"sharded top-k differs with {shards} shards"
    print("check: sharded top-10 == global top-10 for 1, 4, 7 and 32 shards  OK\n")


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1000:
            return f"{nbytes:,.0f} {unit}" if unit == "B" else f"{nbytes:,.1f} {unit}"
        nbytes /= 1000
    return f"{nbytes:,.1f} PB"


def secs(s):
    return f"{s * 1000:.0f} ms" if s < 0.01 else f"{s:.2f} s" if s < 10 else f"{s:,.0f} s"


def hours(sec):
    return f"{sec / 3600:,.1f} h" if sec >= 3600 else f"{sec / 60:,.1f} min"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ram-gb", type=float, default=12, help="ASSUMED RAM of one worker (Colab free ~ 12 GB)")
    ap.add_argument("--docs-per-sec", type=float, default=1000, help="ASSUMED encoding rate of ONE worker")
    ap.add_argument("--serial-fraction", type=float, default=0.02, help="ASSUMED serial+communication fraction f")
    ap.add_argument("--mtbf-days", type=float, default=1000, help="ASSUMED mean time between failures of ONE machine")
    ap.add_argument("--gbps", type=float, default=None, help="use this scan bandwidth instead of measuring it")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    check_sharded_search()

    if args.gbps:
        gbps = args.gbps
        print(f"GIVEN scan bandwidth: {gbps:.1f} GB/s\n")
    else:
        gbps = measure_scan_gbps()
        print(f"MEASURED on this machine: scan bandwidth ~ {gbps:.1f} GB/s  (float32 matrix @ vector)\n")

    # ---- 1. the scale table --------------------------------------------------
    print(f"1) Embedding matrix, {DIM}-d float32, one brute-force query "
          f"(RAM per worker assumed {args.ram_gb:g} GB)")
    print(f"{'docs':>12} {'matrix':>10} {'x RAM':>7} {'fits?':>6} {'one scan @1 worker':>20}")
    for n in (20_000, 1_000_000, 10_000_000, 100_000_000):
        size = n * DIM * FP32
        ratio = size / (args.ram_gb * 1e9)
        scan = size / (gbps * 1e9)
        note = secs(scan) if ratio <= 1 else f"({secs(scan)} if it fit)"
        print(f"{n:>12,} {human(size):>10} {ratio:>6.1f}x {'yes' if ratio <= 1 else 'NO':>6} {note:>20}")

    # ---- 2. what moves per query --------------------------------------------
    n, size = 100_000_000, 100_000_000 * DIM * FP32
    print(f"\n2) Move the query, not the data  (100M docs, k = {args.k})")
    print(f"{'shards':>7} {'rows/shard':>12} {'shard size':>11} {'scan time':>10} "
          f"{'query out':>10} {'results in':>11} {'moved / scanned':>16}")
    for w in (32, 128):
        q_out = w * DIM * FP32
        r_in = w * args.k * (ID_BYTES + SCORE_BYTES)
        print(f"{w:>7} {n // w:>12,} {human(size / w):>11} {secs(size / w / (gbps * 1e9)):>10} "
              f"{human(q_out):>10} {human(r_in):>11} {(q_out + r_in) / size:>16.1e}")
    print("   (scan time = shard size / measured bandwidth; adds a network round trip of ~ms and the slowest shard)")

    # ---- 3. indexing the corpus ---------------------------------------------
    f, r = args.serial_fraction, args.docs_per_sec
    print(f"\n3) Encoding 100M documents. ASSUMED: {r:g} docs/s per worker, serial fraction f = {f:.0%}")
    print(f"{'workers':>8} {'ideal':>10} {'with f':>10} {'speedup':>8}")
    for w in (1, 8, 32, 128):
        ideal = n / (r * w)
        s = 1 / (f + (1 - f) / w)
        print(f"{w:>8} {hours(ideal):>10} {hours(n / r / s):>10} {s:>7.1f}x")
    print(f"   ceiling 1/f = {1 / f:.0f}x : more workers cannot beat it")

    # ---- 4. failures ----------------------------------------------------------
    print(f"\n4) P(at least one failure) = 1 - exp(-workers x hours / MTBF), MTBF ASSUMED {args.mtbf_days:g} days/machine")
    print(f"{'workers':>8} {'job length':>11} {'P(failure)':>11}")
    for w, h in ((1, 1), (32, 1), (128, 8), (1000, 24)):
        p = 1 - math.exp(-w * h / (args.mtbf_days * 24))
        print(f"{w:>8} {h:>9} h {p * 100:>10.2g}%")


if __name__ == "__main__":
    main()
