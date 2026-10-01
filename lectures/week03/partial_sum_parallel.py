"""
Week 3 – Parallel partial sums with lock (correct pattern).
Run: python partial_sum_parallel.py
"""
import numpy as np
import threading
import time

data = np.random.rand(10_000_000)
result = 0.0
lock = threading.Lock()

def partial_sum(start, end):
    global result
    local = np.sum(data[start:end])  # compute locally, no lock
    with lock:
        result += local              # one lock operation

# Parallel version
t0 = time.perf_counter()
threads = []
n_chunks = 4
chunk_size = len(data) // n_chunks
for i in range(n_chunks):
    s, e = i * chunk_size, (i + 1) * chunk_size
    t = threading.Thread(target=partial_sum, args=(s, e))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
t_parallel = time.perf_counter() - t0

# Serial version for comparison
t0 = time.perf_counter()
serial_result = np.sum(data)
t_serial = time.perf_counter() - t0

print(f"Serial sum:   {serial_result:.6f} ({t_serial:.4f} s)")
print(f"Parallel sum: {result:.6f} ({t_parallel:.4f} s)")
print(f"Correct: {np.isclose(result, serial_result)}")