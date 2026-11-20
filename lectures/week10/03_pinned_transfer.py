"""Pinned memory and non_blocking copies: speed, and the two hazards. Needs a CUDA GPU.

    python 03_pinned_transfer.py
    python 03_pinned_transfer.py --mb 64

Part 1 times host-to-device copies of one batch: pageable, pinned, and "pin by hand".
Part 2 shows why non_blocking=True needs care:
  - H2D: change a pinned source after the copy was queued, and the GPU may receive the change.
  - D2H: read the destination before synchronising, and you may read an unfinished copy.
Based on PyTorch's guide "A guide on good usage of non_blocking and pin_memory()"
(https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
"""
import argparse
import statistics
import time

import torch


def time_ms(fn, runs=20, warmup=3):
    """Median wall time of fn() in ms, with a GPU sync before the clock stops."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def part1_speed(mb):
    n = int(mb * 1e6 / 4)
    pageable = torch.randn(n)
    pinned = torch.randn(n).pin_memory()
    rows = [
        ("pageable.to(cuda)", lambda: pageable.to("cuda")),
        ("pinned.to(cuda)", lambda: pinned.to("cuda")),
        ("pinned.to(cuda, non_blocking=True)", lambda: pinned.to("cuda", non_blocking=True)),
        ("pageable.pin_memory().to(cuda)  (by hand)", lambda: pageable.pin_memory().to("cuda")),
    ]
    print(f"Part 1: copying {mb:g} MB host -> device (median of 20 runs, synced)\n")
    for name, fn in rows:
        ms = time_ms(fn)
        print(f"  {name:44s} {ms:8.2f} ms   {mb / ms:6.1f} GB/s")
    print("\n  Pinning by hand costs a host-side copy every time. Let the DataLoader pin\n"
          "  batches in its own thread (pin_memory=True) instead.\n")


def part2_hazards(trials=5):
    print("Part 2: two hazards of non_blocking=True\n")
    n = 1_000_000
    sleep_cycles = int(2e8)                     # keep the GPU busy so the copy waits in the queue

    def h2d(pinned):
        wrong = 0
        for _ in range(trials):
            src = torch.ones(n)
            if pinned:
                src = src.pin_memory()
            torch.cuda._sleep(sleep_cycles)     # the copy below queues behind this
            dst = src.to("cuda", non_blocking=True)
            src.zero_()                         # the host changes the source right away
            torch.cuda.synchronize()
            wrong += int(dst.sum().item() != n)  # 1.0 each if the copy happened before zero_()
        return wrong

    print(f"  H2D, change the source after the call ({trials} trials):")
    print(f"    pageable source: {h2d(False)} wrong   (safe: the copy is staged before returning)")
    print(f"    pinned source:   {h2d(True)} wrong   (the GPU may see the zeros)")

    def d2h(sync_first):
        wrong = 0
        for _ in range(trials):
            x = torch.ones(n, device="cuda")
            torch.cuda._sleep(sleep_cycles)
            y = x * 2                           # result is not ready yet
            host = y.to("cpu", non_blocking=True)
            if sync_first:
                torch.cuda.synchronize()
            wrong += int(host.sum().item() != 2 * n)
        return wrong

    print(f"\n  D2H, read the result ({trials} trials):")
    print(f"    read at once:        {d2h(False)} wrong   (the copy may not have finished)")
    print(f"    synchronize first:   {d2h(True)} wrong")
    print("\n  Rule: after a non_blocking copy to the CPU, synchronize (or wait on an event)\n"
          "  before you touch the data. Never modify a pinned source while a copy may be pending.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mb", type=float, default=77, help="size of the batch in MB (default 77 = 128x3x224x224 float32)")
    p.add_argument("--trials", type=int, default=5, help="trials for each hazard demo (default 5)")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This script needs a CUDA GPU (on Colab: Runtime > Change runtime type > T4 GPU).")
    print(f"{torch.cuda.get_device_name(0)}   torch {torch.__version__}\n")
    part1_speed(a.mb)
    part2_hazards(a.trials)


if __name__ == "__main__":
    main()
