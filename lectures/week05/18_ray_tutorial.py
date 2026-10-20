"""
Week 5 · Ray tutorial, one step at a time.

    python 18_ray_tutorial.py 1        # run one step (1-8)
    python 18_ray_tutorial.py all      # run every step in order

    1  start and stop Ray, look at the resources
    2  tasks: @ray.remote, .remote(), ray.get
    3  objects and dependencies: ray.put, futures as arguments
    4  resources: num_cpus, available_resources
    5  waiting: ray.wait, timeouts, cancel
    6  actors: state, named actors, ray.kill
    7  watch it run: ray.timeline, dashboard
    8  failure: retries (see 15_ray_fault_tolerance.py for the full demo)

Needs:  pip install ray        (add  "ray[default]"  for the dashboard on port 8265)
Each step starts and stops its own Ray, so any step can be run alone.
"""
import logging
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import ray


def start(cpus=2):
    ray.init(num_cpus=cpus, include_dashboard=False, logging_level=logging.ERROR)


# ---------------------------------------------------------------- 1
def step1_init():
    start(2)
    print("initialised   :", ray.is_initialized())
    res = ray.cluster_resources()
    print("cluster CPUs  :", res["CPU"])                  # 2.0: what we asked for, not what the laptop has
    print("nodes         :", len(ray.nodes()))
    ray.shutdown()
    print("initialised   :", ray.is_initialized())


# ---------------------------------------------------------------- 2
@ray.remote
def slow_square(x):
    time.sleep(0.5)
    return x * x


def step2_tasks():
    start(2)
    t0 = time.perf_counter()
    refs = [slow_square.remote(i) for i in range(8)]     # returns at once: these are futures (ObjectRefs)
    print(f"submitted 8 tasks in {time.perf_counter() - t0:.3f} s ->", refs[0])
    results = ray.get(refs)                              # blocks until all are done
    print(f"results {results} after {time.perf_counter() - t0:.2f} s  (8 x 0.5 s on 2 CPUs = 2 s)")
    assert results == [i * i for i in range(8)]
    ray.shutdown()


# ---------------------------------------------------------------- 3
@ray.remote
def make_matrix(n):
    return np.ones((n, n))


@ray.remote
def total(m):
    return float(m.sum())


@ray.remote
def add(a, b):
    return a + b


def step3_objects():
    start(2)
    ref = ray.put(np.arange(5))                           # store a value once, get a reference
    print("ray.get(ref)      :", ray.get(ref))
    m = make_matrix.remote(1000)                          # the result stays in the object store
    s = total.remote(m)                                   # passing a ref = declaring a dependency
    print("sum of 1000x1000  :", ray.get(s), " (only a float came back to the driver)")
    assert ray.get(add.remote(add.remote(1, 2), add.remote(3, 4))) == 10    # futures as arguments
    ray.shutdown()


# ---------------------------------------------------------------- 4
@ray.remote(num_cpus=1)
def one_core(i):
    time.sleep(0.5)
    return i


@ray.remote(num_cpus=2)
def two_cores(i):
    time.sleep(0.5)
    return i


def step4_resources():
    start(2)
    t0 = time.perf_counter()
    refs = [one_core.remote(i) for i in range(4)]
    time.sleep(0.2)
    print("free CPUs while 2 tasks run:", ray.available_resources().get("CPU", 0))
    ray.get(refs)
    print(f"4 tasks x 1 CPU : {time.perf_counter() - t0:.2f} s   (2 at a time)")
    t0 = time.perf_counter()
    ray.get([two_cores.remote(i) for i in range(2)])
    print(f"2 tasks x 2 CPUs: {time.perf_counter() - t0:.2f} s   (1 at a time)")
    t0 = time.perf_counter()
    ray.get([one_core.options(num_cpus=2).remote(i) for i in range(2)])   # override per call
    print(f"options(num_cpus=2): {time.perf_counter() - t0:.2f} s")
    print("resources are logical labels: Ray never runs more tasks than CPUs you declared")
    ray.shutdown()


# ---------------------------------------------------------------- 5
@ray.remote
def work(seconds):
    time.sleep(seconds)
    return seconds


def step5_waiting():
    start(2)
    refs = [work.remote(s) for s in (0.2, 0.4, 5.0, 0.3)]
    pending = list(refs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=1.5)   # next finished, or give up after 1.5 s
        if not ready:
            print("timeout: giving up on", len(pending), "straggler(s)")
            for r in pending:
                ray.cancel(r)
            break
        print("finished:", ray.get(ready[0]))
    try:
        ray.get(refs[2], timeout=5)
    except ray.exceptions.RayError as e:
        print("cancelled task ->", type(e).__name__)
    ray.shutdown()


# ---------------------------------------------------------------- 6
@ray.remote
class Counter:
    def __init__(self):
        self.n = 0                                        # state lives inside the actor process

    def incr(self):
        self.n += 1
        return self.n


def step6_actors():
    start(2)
    c = Counter.remote()
    print("calls:", [ray.get(c.incr.remote()) for _ in range(3)])      # 1, 2, 3: state is kept
    named = Counter.options(name="shared-counter").remote()
    same = ray.get_actor("shared-counter")                # any code in this job can find it by name
    ray.get(named.incr.remote())
    print("via the name:", ray.get(same.incr.remote()))   # 2
    ray.kill(c)                                           # actors live until you kill them (or the job ends)
    ray.shutdown()


# ---------------------------------------------------------------- 7
def step7_watch():
    start(2)
    ray.get([slow_square.remote(i) for i in range(6)])
    ray.timeline(filename="ray-timeline.json")
    print("wrote ray-timeline.json  (open chrome://tracing or https://ui.perfetto.dev and load it)")
    print("dashboard: pip install 'ray[default]', then ray.init(include_dashboard=True) -> http://127.0.0.1:8265")
    ray.shutdown()


# ---------------------------------------------------------------- 8
@ray.remote(max_retries=2, retry_exceptions=True)
def flaky(path):
    n = int(open(path).read() or 0) + 1
    open(path, "w").write(str(n))
    if n < 3:
        raise ValueError(f"attempt {n} failed")
    return f"succeeded on attempt {n}"


def step8_failure():
    import tempfile
    start(2)
    path = os.path.join(tempfile.mkdtemp(), "count")
    open(path, "w").write("0")
    print(ray.get(flaky.remote(path)))                    # retried twice, then succeeds
    print("defaults: a dead worker is retried (max_retries=3); a raised exception is NOT (retry_exceptions=False)")
    ray.shutdown()


STEPS = {1: step1_init, 2: step2_tasks, 3: step3_objects, 4: step4_resources,
         5: step5_waiting, 6: step6_actors, 7: step7_watch, 8: step8_failure}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = list(STEPS) if arg == "all" else [int(arg)]
    for k in todo:
        print(f"\n=== step {k}: {STEPS[k].__name__} ===")
        STEPS[k]()
