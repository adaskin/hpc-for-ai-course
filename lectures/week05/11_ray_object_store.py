"""
Week 5 · Ray's object store: put big read-only data once, pass a reference.

  1. by value vs by reference (20 tasks, one 16 MB array)
  2. what ray.get returns for a NumPy array: a read-only, shared-memory view
  3. results can stay in the store: pass a future between tasks, only a float comes back

Needs:  pip install ray
Run:    python 11_ray_object_store.py
"""
import logging

import numpy as np
import ray

from corpus import Timer


@ray.remote
def score(matrix, i):
    return float(matrix[i::20].sum())             # a small answer from a big input


@ray.remote
def make_matrix(n_rows):
    return np.ones((n_rows, 1_000))


@ray.remote
def total(m):
    return float(m.sum())


if __name__ == "__main__":
    ray.init(num_cpus=2, include_dashboard=False, logging_level=logging.ERROR)
    matrix = np.random.default_rng(0).random((2_000, 1_000))      # 16 MB
    ray.get(score.remote(matrix, 0))                              # warm-up

    # ---- 1. by value vs by reference ----
    with Timer() as t_val:
        by_value = ray.get([score.remote(matrix, i) for i in range(20)])
    ref = ray.put(matrix)                                         # serialised ONCE
    with Timer() as t_ref:
        by_ref = ray.get([score.remote(ref, i) for i in range(20)])
    assert by_value == by_ref
    print(f"20 tasks, 16 MB input: by value {t_val.s:.2f} s | ray.put once {t_ref.s:.2f} s")

    # ---- 2. read-only, zero-copy arrays ----
    arr = ray.get(ref)
    print("array from the object store is writeable:", arr.flags.writeable)
    try:
        arr[0, 0] = -1.0
        print("(this Ray version returned a writeable copy)")
    except ValueError as e:
        print("writing refused:", e)
    mine = arr.copy()                                             # copy if you must modify
    mine[0, 0] = -1.0

    # ---- 3. results stay in the store ----
    m_ref = make_matrix.remote(5_000)                             # 40 MB, built on a worker
    s = ray.get(total.remote(m_ref))                              # worker -> worker; only a float returns
    assert s == 5_000 * 1_000
    print("sum of a 40 MB matrix that never visited the driver:", s)

    ray.shutdown()
