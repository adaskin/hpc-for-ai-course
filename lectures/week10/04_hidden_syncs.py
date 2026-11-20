"""Hidden syncs: .item() every step vs every N steps, and where the time really goes.

    python 04_hidden_syncs.py
    python 04_hidden_syncs.py --every 1 10 1000

Each run uses the same fast loader, so any difference comes from how often the loop
calls loss.item(), which waits for the GPU. On a CPU-only machine there is no GPU to wait
for, so the times are about equal; run it on a GPU to see the effect.

The table also splits time into "data wait" (host time inside next(loader)) and "step"
(forward, backward, update). See pipeline_utils.run_epochs for the exact timing recipe.
"""
import argparse
import os

import torch

from pipeline_utils import BatchedDataset, get_device, make_loader, passthrough, run_epochs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--every", type=int, nargs="+", default=[1, 10, 1000],
                   help="call loss.item() every N steps (default 1 10 1000)")
    p.add_argument("--size", type=int, default=4096, help="samples (default 4096)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    a = p.parse_args()

    dev = get_device()
    print(f"device: {dev}   torch {torch.__version__}")
    ds = BatchedDataset(a.size, delay=0.0005)
    loader = make_loader(ds, a.batch, workers=a.workers, pin=dev.type == "cuda", persistent=True,
                         collate_fn=passthrough)
    run_epochs(loader, epochs=1, device=dev)                      # warm-up

    print(f"\n{'loss.item() every':>18s} {'seconds':>8s} {'data wait':>10s} {'step':>8s}")
    for n in a.every:
        r = run_epochs(loader, epochs=a.epochs, item_every=n, device=dev,
                       non_blocking=dev.type == "cuda")
        print(f"{n:>18d} {r['seconds']:8.2f} {r['data_wait']:9.2f}s {r['step']:7.2f}s")
    if dev.type == "cpu":
        print("\n(CPU only: nothing to wait for, so the rows are alike. Try this on a GPU.)")


if __name__ == "__main__":
    main()
