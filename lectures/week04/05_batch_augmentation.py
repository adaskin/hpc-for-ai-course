"""
Week 4 Part II – Batch augmentation: threads work (NumPy releases GIL).
Run: python 05_batch_augmentation.py
"""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def augment_image(img: np.ndarray) -> np.ndarray:
    """Random horizontal flip + brightness jitter. All NumPy."""
    if np.random.rand() > 0.5:
        img = img[:, ::-1, :].copy()  # horizontal flip
    img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
    return img


def main():
    n_images = 32
    n_workers = 8
    batch = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(n_images)]

    print(f"Augmenting {n_images} images with {n_workers} threads\n")

    # Serial
    t0 = time.perf_counter()
    serial = [augment_image(img) for img in batch]
    t_serial = time.perf_counter() - t0

    # Threads (NumPy releases GIL → speedup!)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        threaded = list(ex.map(augment_image, batch))
    t_threads = time.perf_counter() - t0

    print(f"Serial:   {t_serial:.4f} s")
    print(f"Threads:  {t_threads:.4f} s  (speedup: {t_serial/t_threads:.1f}×)")
    print(f"\nThreads work here because NumPy operations release the GIL.")


if __name__ == "__main__":
    main()