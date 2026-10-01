"""
Week 3 – Barrier: all threads must finish phase 1 before phase 2.
Run: python barrier_sync.py
"""
import threading
import time
import random

N_THREADS = 4
barrier = threading.Barrier(N_THREADS)

def worker(thread_id):
    # Phase 1: load data (different time per thread)
    load_time = random.uniform(0.5, 2.0)
    print(f"  Thread {thread_id}: loading data ({load_time:.1f}s)...")
    time.sleep(load_time)
    print(f"  Thread {thread_id}: phase 1 done, waiting at barrier")

    barrier.wait()  # ALL threads must reach here

    # Phase 2: compute (starts simultaneously for all)
    print(f"  Thread {thread_id}: phase 2 START")
    time.sleep(0.5)
    print(f"  Thread {thread_id}: phase 2 done")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("All phases complete.")