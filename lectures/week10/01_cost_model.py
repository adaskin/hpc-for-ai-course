"""Cost model for an input pipeline. Pure Python: no GPU and no PyTorch needed.

    python 01_cost_model.py
    python 01_cost_model.py --t-data 30 --t-gpu 50 --batch 128 --per-sample 10 --cores 2

serial step      = t_data + t_gpu              (the GPU waits while the loader works)
overlapped step  = max(t_data, t_gpu)          (the loader prepares batch k+1 during step k)
GPU busy share   = t_gpu / step
workers needed   = (samples/s the GPU consumes) / (samples/s one worker delivers)
"""
import argparse
import math


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--t-data", type=float, default=30, help="ms to prepare one batch (default 30)")
    p.add_argument("--t-gpu", type=float, default=50, help="ms of GPU work per step (default 50)")
    p.add_argument("--batch", type=int, default=128, help="samples per batch (default 128)")
    p.add_argument("--per-sample", type=float, default=10, help="ms CPU time per sample in one worker (default 10)")
    p.add_argument("--cores", type=int, default=2, help="CPU cores you have (default 2)")
    p.add_argument("--samples", type=int, default=50_000, help="samples per epoch (default 50000)")
    a = p.parse_args()

    serial = a.t_data + a.t_gpu
    overlapped = max(a.t_data, a.t_gpu)
    print("one step")
    print(f"  serial      {serial:7.1f} ms   GPU busy {100 * a.t_gpu / serial:5.1f} %")
    print(f"  overlapped  {overlapped:7.1f} ms   GPU busy {100 * a.t_gpu / overlapped:5.1f} %")
    bound = "input-bound" if a.t_data > a.t_gpu else "compute-bound"
    print(f"  with overlap the job is {bound}: the slower side sets the pace")

    need = a.batch / (a.t_gpu / 1000)                 # samples/s the GPU can consume
    one = 1000 / a.per_sample                         # samples/s from one worker
    workers = math.ceil(need / one)
    print("\nworker sizing")
    print(f"  GPU consumes   {need:8.0f} samples/s   (batch {a.batch} per {a.t_gpu:g} ms)")
    print(f"  one worker     {one:8.0f} samples/s   ({a.per_sample:g} ms per sample)")
    print(f"  workers needed {workers:8d}")
    got = min(workers, a.cores) * one
    if workers > a.cores:
        print(f"  you have {a.cores} cores -> at most {got:.0f} samples/s: more workers will not help.")
        print("  Reduce the work per sample instead (batched reads, caching, smaller dtypes).")
    epoch_fast = a.samples / need
    epoch_real = a.samples / min(need, got)
    print(f"  epoch of {a.samples} samples: {epoch_fast:.1f} s if the GPU is fed, {epoch_real:.1f} s with {a.cores} cores")


if __name__ == "__main__":
    main()
