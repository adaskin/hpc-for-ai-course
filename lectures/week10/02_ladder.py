"""The optimisation ladder, one step at a time, on a synthetic dataset.

Follows PyTorch's tutorial "Data Loading Optimization in PyTorch": start from the slowest
loader and add one change per step, recording the speedup of each step and the total.

    python 02_ladder.py
    python 02_ladder.py --workers 2 --epochs 3 --delay 0.005

Steps that need a GPU (pin_memory, overlapped copies) are skipped on a CPU-only machine.
Every step must see exactly the same data: the script compares a checksum with the baseline.
Run it from this directory.
"""
import argparse
import os

import torch

from pipeline_utils import (BatchedDataset, SyntheticDataset, get_device, make_loader,
                            passthrough, run_epochs)


def main():
    cpus = os.cpu_count() or 1
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=512, help="samples in the dataset (default 512)")
    p.add_argument("--batch", type=int, default=32, help="batch size (default 32)")
    p.add_argument("--delay", type=float, default=0.005, help="seconds of fixed cost per call (default 0.005)")
    p.add_argument("--workers", type=int, default=min(4, cpus), help=f"worker processes (default min(4, {cpus} cores))")
    p.add_argument("--epochs", type=int, default=3, help="passes over the data (default 3)")
    a = p.parse_args()

    dev = get_device()
    cuda = dev.type == "cuda"
    print(f"device: {dev}   cpu cores: {cpus}   torch {torch.__version__}")
    print(f"{a.size} samples, batch {a.batch}, {a.delay * 1000:g} ms fixed cost per call, "
          f"{a.workers} workers, {a.epochs} epochs\n")

    per_sample = SyntheticDataset(a.size, delay=a.delay)
    batched = BatchedDataset(a.size, delay=a.delay)
    W = a.workers

    # name, loader, run_epochs options (None = skipped, with the reason)
    steps = [
        ("baseline: num_workers=0", make_loader(per_sample, a.batch), {}),
        (f"num_workers={W}", make_loader(per_sample, a.batch, workers=W), {}),
        ("pin_memory=True", make_loader(per_sample, a.batch, workers=W, pin=True), {"non_blocking": True})
        if cuda else ("pin_memory=True", None, "needs a GPU"),
        ("persistent_workers=True", make_loader(per_sample, a.batch, workers=W, pin=cuda, persistent=True),
         {"non_blocking": cuda}),
        ("copy on a side stream", make_loader(per_sample, a.batch, workers=W, pin=True, persistent=True),
         {"prefetch": True}) if cuda else ("copy on a side stream", None, "needs a GPU"),
        ("__getitems__ (batched call)", make_loader(batched, a.batch, workers=W, pin=cuda, persistent=True,
                                                    collate_fn=passthrough),
         {"non_blocking": cuda}),
    ]

    base = prev = None
    base_check = None
    print(f"{'step':32s} {'seconds':>8s} {'vs prev':>8s} {'total':>8s} {'data wait':>10s}")
    for name, loader, opts in steps:
        if loader is None:
            print(f"{name:32s} {'skipped: ' + opts:>38s}")
            continue
        r = run_epochs(loader, epochs=a.epochs, **opts)
        if base is None:
            base, base_check = r["seconds"], r["checksum"]
        assert abs(r["checksum"] - base_check) <= 1e-3 * abs(base_check) + 1e-3, \
            f"{name}: the data differs from the baseline"
        vs_prev = f"{prev / r['seconds']:.2f}x" if prev else "-"
        print(f"{name:32s} {r['seconds']:8.2f} {vs_prev:>8s} {base / r['seconds']:7.2f}x "
              f"{100 * r['data_wait'] / r['seconds']:9.0f}%")
        prev = r["seconds"]
    print("\nAll steps saw the same data (checksum matches the baseline).")


if __name__ == "__main__":
    main()
