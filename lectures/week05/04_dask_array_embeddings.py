"""
Week 5 · Dask Array on an AI-shaped problem: a tall-skinny embedding matrix X (N docs x d dims).

- Gram matrix X^T X: every chunk contributes a small d x d block, merged by a cheap reduce.
- Randomised SVD of X with dask.array.linalg.svd_compressed.
Both are checked against NumPy.

Message: if X fits in RAM, NumPy (BLAS, already multi-threaded) is hard to beat.
Dask earns its keep when X does NOT fit, or when the work must go beyond one machine.

Run:  python 04_dask_array_embeddings.py
"""
import numpy as np
import dask.array as da

from corpus import Timer

N, D, RANK = 100_000, 256, 20

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # low-rank structure + small noise, like real embeddings: the top singular values stand out
    X = (rng.standard_normal((N, RANK)) @ rng.standard_normal((RANK, D))
         + 0.01 * rng.standard_normal((N, D))).astype(np.float32)
    Xd = da.from_array(X, chunks=(10_000, D))                  # chunk along rows only
    print(f"X: {X.nbytes / 1e6:.0f} MB in {Xd.numblocks[0]} row chunks of {Xd.chunksize[0]} rows")

    # ---- Gram matrix ----
    with Timer() as t_np:
        G_ref = X.T @ X
    with Timer() as t_da:
        G = (Xd.T @ Xd).compute()
    err = np.linalg.norm(G - G_ref) / np.linalg.norm(G_ref)
    print(f"Gram matrix        NumPy {t_np.s:.3f} s | Dask {t_da.s:.3f} s | relative error {err:.1e}")
    assert err < 1e-4

    # ---- top-RANK singular values ----
    with Timer() as t_np:
        s_ref = np.linalg.svd(X, compute_uv=False)[:RANK]
    with Timer() as t_da:
        u, s, v = da.linalg.svd_compressed(Xd, k=RANK, n_power_iter=2, seed=0)
        s = s.compute()
    rel = float(np.max(np.abs(s - s_ref) / s_ref))
    print(f"top-{RANK} singular values  NumPy {t_np.s:.2f} s | Dask randomised {t_da.s:.2f} s "
          f"| max relative error {rel:.1e}")
    assert rel < 1e-2
