"""
Week 3 – Race condition demo: unsynchronised counter.
Run: python race_condition_counter.py
Expected: 20,000,000. Actual: less (varies each run).
"""
import threading

counter = 0

def increment():
    global counter
    for _ in range(10_000_000):
        counter += 1  # NOT atomic!

threads = [threading.Thread(target=increment) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Expected: 20,000,000")
print(f"Actual:   {counter:,}")
print(f"Lost:     {20_000_000 - counter:,}")
