"""
Week 3 – Producer-consumer with queue.Queue (thread-safe).
Run: python producer_consumer_queue.py
"""
import queue
import threading
import time
import random

q = queue.Queue(maxsize=5)  # bounded buffer

def producer(name, n_items):
    for i in range(n_items):
        item = f"{name}_item_{i}"
        q.put(item)  # blocks if queue full
        print(f"  [{name}] Produced: {item}")
        time.sleep(random.uniform(0.05, 0.2))
    q.put(None)  # sentinel

def consumer(name):
    while True:
        item = q.get()  # blocks if queue empty
        if item is None:
            q.put(None)  # pass sentinel to other consumers
            break
        print(f"  [{name}] Consumed: {item}")
        time.sleep(random.uniform(0.1, 0.3))

producers = [threading.Thread(target=producer, args=(f"P{i}", 5)) for i in range(2)]
consumers = [threading.Thread(target=consumer, args=(f"C{i}",)) for i in range(2)]

for t in producers + consumers:
    t.start()
for t in producers + consumers:
    t.join()

print("All done.")