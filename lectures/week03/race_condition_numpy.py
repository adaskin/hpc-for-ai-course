"""
Week 3 – Race condition on a NumPy array (AI context: gradient accumulation).
Run: python race_condition_numpy.py
"""
import numpy as np
import threading

# Simulate a shared gradient accumulator
grad_accumulator = np.zeros(1000)
lock = threading.Lock()  # try with and without

def accumulate(grads, use_lock=False):
    global grad_accumulator
    for i in range(len(grads)):
        if use_lock:
            with lock:
                grad_accumulator[i] += grads[i]
        else:
            grad_accumulator[i] += grads[i]  # race!

# Run WITHOUT lock
grad_accumulator[:] = 0
g1 = np.ones(1000)
g2 = np.ones(1000)

t1 = threading.Thread(target=accumulate, args=(g1, False))
t2 = threading.Thread(target=accumulate, args=(g2, False))
t1.start(); t2.start(); t1.join(); t2.join()

expected = g1 + g2
errors = np.abs(grad_accumulator - expected)
print(f"WITHOUT lock: max error = {errors.max():.4f}, "
      f"elements wrong = {(errors > 1e-10).sum()}")

# Run WITH lock
grad_accumulator[:] = 0
t1 = threading.Thread(target=accumulate, args=(g1, True))
t2 = threading.Thread(target=accumulate, args=(g2, True))
t1.start(); t2.start(); t1.join(); t2.join()

errors = np.abs(grad_accumulator - expected)
print(f"WITH lock:    max error = {errors.max():.4f}, "
      f"elements wrong = {(errors > 1e-10).sum()}")