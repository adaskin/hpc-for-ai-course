"""
Week 5 · ray.wait: handle results as they finish, set a timeout, drop a straggler.

ray.get(refs) waits for the SLOWEST task. ray.wait lets you act on the first
finished ones (Ray's counterpart of concurrent.futures.as_completed from Week 4).

Needs:  pip install ray
Run:    python 12_ray_wait.py
"""
import logging
import time

import ray

from corpus import Timer

DURATIONS = [0.3, 0.5, 0.2, 4.0, 0.4, 0.3, 0.2, 0.1]          # task 3 is a straggler
STRAGGLER = 3


@ray.remote
def work(i, seconds):
    time.sleep(seconds)
    return i


def submit():
    return [work.remote(i, s) for i, s in enumerate(DURATIONS)]


if __name__ == "__main__":
    ray.init(num_cpus=2, include_dashboard=False, logging_level=logging.ERROR)
    ray.get(work.remote(-1, 0))                                # warm-up

    # ---- 1. ray.get waits for the straggler ----
    with Timer() as t:
        ray.get(submit())
    print(f"ray.get(all): {t.s:.1f} s (the straggler sets the pace)\n")

    # ---- 2. process results as they finish ----
    pending, order = submit(), []
    t0 = time.perf_counter()
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        order.append(ray.get(ready[0]))
        print(f"  t={time.perf_counter() - t0:4.1f} s  finished task {order[-1]}")
    assert order[-1] == STRAGGLER
    print()

    # ---- 3. timeout, then give up on the straggler ----
    refs = submit()
    ready, pending = ray.wait(refs, num_returns=len(refs), timeout=3.0)
    print(f"{len(ready)} of {len(refs)} tasks done within 3 s; cancelling {len(pending)}")
    for r in pending:
        ray.cancel(r)
    done = sorted(ray.get(ready))
    assert done == [i for i in range(len(DURATIONS)) if i != STRAGGLER]
    print("results kept:", done)

    ray.shutdown()
