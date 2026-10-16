"""
Week 5 · Ray actors: stateful workers. Pay an expensive set-up once, then serve many calls.

A stateless task has to rebuild its "model" on every call; an actor keeps it in memory.
(The price: if an actor dies, its state is gone: see 15_ray_fault_tolerance.py.)

Needs:  pip install ray
Run:    python 14_ray_actors.py
"""
import logging
import time

import numpy as np
import ray
from ray.util import ActorPool

from corpus import Timer


def load_model():
    time.sleep(1.0)                                   # pretend: read weights, build an index, ...
    return np.random.default_rng(0).standard_normal((1_000, 64))


@ray.remote
def predict_task(x):
    W = load_model()                                  # paid on EVERY call
    return float((W @ x).sum())


@ray.remote
class Predictor:
    def __init__(self):
        self.W = load_model()                         # paid once per actor
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        return float((self.W @ x).sum())

    def n_calls(self):
        return self.calls


if __name__ == "__main__":
    ray.init(num_cpus=2, include_dashboard=False, logging_level=logging.ERROR)
    xs = [np.random.default_rng(i).standard_normal(64) for i in range(8)]

    with Timer() as t_task:
        by_task = ray.get([predict_task.remote(x) for x in xs])     # 8 x 1 s of loading on 2 CPUs

    actors = [Predictor.remote() for _ in range(2)]
    with Timer() as t_actor:                                          # includes the actors' start-up
        pool = ActorPool(actors)
        by_actor = list(pool.map(lambda a, x: a.predict.remote(x), xs))

    assert np.allclose(by_task, by_actor)
    calls = ray.get([a.n_calls.remote() for a in actors])
    assert sum(calls) == len(xs)
    print(f"8 predictions: stateless tasks {t_task.s:.1f} s | 2 actors {t_actor.s:.1f} s")
    print(f"calls served per actor: {calls}")

    for a in actors:
        ray.kill(a)
    ray.shutdown()
