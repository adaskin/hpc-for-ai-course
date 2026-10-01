
"""
Week 4 Part II – Hyperparameter sweep with ProcessPoolExecutor.
Run: python 03_hyperparam_sweep.py

Simulates running independent training experiments in parallel.
"""
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed


def train_experiment(config: dict) -> dict:
    """Simulate a training run (2 seconds). Returns val_loss."""
    time.sleep(2)  # pretend this is a real training run
    val_loss = random.uniform(0.1, 0.9)
    return {"config": config, "val_loss": val_loss, "epochs": 10}


def main():
    configs = [
        {"lr": lr, "batch_size": bs, "layers": L}
        for lr in [1e-3, 1e-4]
        for bs in [32, 64]
        for L in [4, 8]
    ]  # 8 configs

    print(f"Running {len(configs)} experiments with 4 workers")
    print(f"Each takes ~2s. Serial would be ~16s.\n")

    t0 = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(train_experiment, c): c for c in configs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            c = result["config"]
            print(f"  ✓ lr={c['lr']}, bs={c['batch_size']}, layers={c['layers']} "
                  f"→ val_loss={result['val_loss']:.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.2f} s (vs ~16s serial)")

    # Find best config
    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nBest config: {best['config']} → loss={best['val_loss']:.3f}")


if __name__ == "__main__":
    main()