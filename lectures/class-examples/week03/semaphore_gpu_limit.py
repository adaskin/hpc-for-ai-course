"""
Week 3 – Semaphore: limit concurrent GPU jobs to 2.
Run: python semaphore_gpu_limit.py
"""
import threading
import time

gpu_sem = threading.Semaphore(2)  # only 2 concurrent GPU jobs

def train_model(model_id):
    print(f"  Model {model_id}: waiting for GPU...")
    with gpu_sem:
        print(f"  Model {model_id}: TRAINING on GPU")
        time.sleep(2)  # simulate training
        print(f"  Model {model_id}: done, releasing GPU")

threads = [threading.Thread(target=train_model, args=(i,)) for i in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("All models trained.")