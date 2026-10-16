"""
Week 5 · Who does the arithmetic?  Workers x BLAS threads.

Dask and Ray only coordinate. The arithmetic inside each task is done by NumPy -> BLAS,
and BLAS may start its OWN threads. With W worker processes and T BLAS threads each,
the machine runs W x T threads. If W x T > cores, they fight for the cores
("oversubscription").

This script uses a plain process pool so it needs no extra library, but the effect is
exactly the same inside Dask or Ray workers. For each setting of T it starts a fresh
child interpreter (BLAS reads its thread count at import time).

Run:  python 09_blas_oversubscription.py
Rule of thumb on a cluster:  workers x BLAS threads  <=  cores.
"""
import os
import subprocess
import sys
import time

N_TASKS, SIZE, WORKERS = 8, 1500, os.cpu_count() or 2
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def task(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((SIZE, SIZE))
    return float(np.linalg.norm(a @ a))          # one big matmul -> BLAS


def child():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(WORKERS) as pool:
        list(pool.map(task, range(WORKERS)))      # warm-up: start workers, import NumPy
        t0 = time.perf_counter()
        list(pool.map(task, range(N_TASKS)))
        print(f"{time.perf_counter() - t0:.2f}")


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
        sys.exit()
    cores = os.cpu_count()
    print(f"{cores} cores, {WORKERS} worker processes, {N_TASKS} matmul tasks of {SIZE}x{SIZE}\n")
    print(f"{'BLAS threads per worker':<26}{'total threads':>14}{'time (s)':>10}")
    for t in (1, cores, 2 * cores):
        env = dict(os.environ, **{v: str(t) for v in THREAD_VARS})
        out = subprocess.run([sys.executable, __file__, "--child"], env=env,
                             capture_output=True, text=True, check=True).stdout.strip()
        print(f"{t:<26}{WORKERS * t:>14}{float(out):>10.2f}")
    print("\nExpect: 1 thread per worker is as fast or faster than more; "
          "more threads than cores does not help.")
