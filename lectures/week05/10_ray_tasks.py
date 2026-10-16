"""
Week 5 · Ray tasks: .remote() returns a future at once; ray.get() waits for the value.
Passing a future to another task builds the dependency graph while the program runs
(a dynamic graph).

Needs:  pip install ray
Run:    python 10_ray_tasks.py
"""
import logging
import time

import ray

from corpus import Timer


@ray.remote
def slow_square(x):
    time.sleep(0.5)
    return x * x


@ray.remote
def add(a, b):
    return a + b


if __name__ == "__main__":
    ray.init(num_cpus=2, include_dashboard=False, logging_level=logging.ERROR)
    ray.get(slow_square.remote(0))                       # warm-up: start the worker processes

    # ---- 1. submitting is instant; the work happens in the background ----
    with Timer() as t_submit:
        refs = [slow_square.remote(i) for i in range(8)]
    with Timer() as t_get:
        results = ray.get(refs)
    assert results == [i * i for i in range(8)]
    print(f"submit 8 tasks: {t_submit.s * 1000:.1f} ms | ray.get: {t_get.s:.2f} s "
          f"(8 x 0.5 s on 2 CPUs is about 2 s)")

    # ---- 2. anti-pattern: ray.get inside the loop serialises everything ----
    with Timer() as t_bad:
        bad = [ray.get(slow_square.remote(i)) for i in range(4)]
    with Timer() as t_good:
        good = ray.get([slow_square.remote(i) for i in range(4)])
    assert bad == good
    print(f"ray.get inside the loop: {t_bad.s:.2f} s | submit all, then get: {t_good.s:.2f} s")

    # ---- 3. futures as arguments: Ray resolves the dependencies (a tree reduce) ----
    level = [slow_square.remote(i) for i in range(8)]
    while len(level) > 1:
        nxt = [add.remote(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    answer = ray.get(level[0])
    assert answer == sum(i * i for i in range(8))
    print("tree reduce of 8 squares:", answer)

    ray.shutdown()
