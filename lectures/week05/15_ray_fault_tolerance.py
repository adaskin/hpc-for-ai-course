"""
Week 5 · Ray fault tolerance: what gets retried, and what does not.

  A. a worker process dies mid-task      -> retried automatically (max_retries, default 3)
  B. the task raises a Python exception  -> NOT retried unless retry_exceptions=True
  C. an actor dies and is restarted      -> restarted, but its in-memory state is gone

The crash is simulated with os._exit(1) inside the task. Ray prints a few warnings
about the dead worker; that is expected.

Needs:  pip install ray
Run:    python 15_ray_fault_tolerance.py
"""
import logging
import os
import tempfile
import time

import ray


@ray.remote
def crash_once(marker):
    """The first call kills its own worker process, like a node dying."""
    if not os.path.exists(marker):
        open(marker, "w").close()
        os._exit(1)
    return "survived"


@ray.remote
def flaky(counter_file):
    """Raises ValueError on attempts 1 and 2, succeeds on attempt 3."""
    with open(counter_file) as f:
        n = int(f.read() or 0) + 1
    with open(counter_file, "w") as f:
        f.write(str(n))
    if n < 3:
        raise ValueError(f"attempt {n} failed")
    return f"succeeded on attempt {n}"


@ray.remote(max_restarts=1)
class Counter:
    def __init__(self):
        self.n = 0

    def incr(self):
        self.n += 1
        return self.n


if __name__ == "__main__":
    ray.init(num_cpus=2, include_dashboard=False, logging_level=logging.ERROR)
    tmp = tempfile.mkdtemp()

    # ---- A. worker crash ----
    print("A. worker crash")
    print("   with default retries:", ray.get(crash_once.remote(os.path.join(tmp, "a"))))
    try:
        ray.get(crash_once.options(max_retries=0).remote(os.path.join(tmp, "b")))
    except ray.exceptions.RayError as e:
        print("   with max_retries=0 :", type(e).__name__)

    # ---- B. application exception ----
    print("B. application exception")
    path = os.path.join(tmp, "count")
    with open(path, "w") as f:
        f.write("0")
    try:
        ray.get(flaky.remote(path))
    except ray.exceptions.RayTaskError as e:
        print("   default            : raised", type(e.cause).__name__, "- not retried")
    with open(path, "w") as f:
        f.write("0")
    result = ray.get(flaky.options(retry_exceptions=True, max_retries=3).remote(path))
    print("   retry_exceptions   :", result)
    assert result == "succeeded on attempt 3"

    # ---- C. actor restart loses state ----
    print("C. actor restart")
    c = Counter.remote()
    before = [ray.get(c.incr.remote()) for _ in range(3)]
    ray.kill(c, no_restart=False)                      # the actor process dies; Ray restarts it
    after = None
    for _ in range(20):
        try:
            after = ray.get(c.incr.remote())
            break
        except ray.exceptions.RayError:
            time.sleep(0.5)                            # still restarting
    print(f"   counter before the crash: {before[-1]} | first value after restart: {after}")
    assert after == 1                                  # __init__ ran again: the state was lost

    ray.shutdown()
