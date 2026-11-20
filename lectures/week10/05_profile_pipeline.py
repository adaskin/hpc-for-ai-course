"""Profile a training loop and label the data wait, then open the trace in Perfetto.

    python 05_profile_pipeline.py                 # slow loader (num_workers=0)
    python 05_profile_pipeline.py --workers 4 --batched --out fast_trace.json

Open the JSON at https://ui.perfetto.dev (or chrome://tracing). Look for the "data_wait"
blocks: gaps between GPU kernels line up with them when the job is input-bound.
Based on the PyTorch Profiler recipe
(https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
"""
import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from pipeline_utils import (BatchedDataset, SyntheticDataset, get_device, make_loader,
                            make_model, passthrough, sync)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--batched", action="store_true", help="use __getitems__ (one call per batch)")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--out", default="week10_trace.json")
    a = p.parse_args()

    dev = get_device()
    ds = (BatchedDataset if a.batched else SyntheticDataset)(512, delay=0.005)
    loader = make_loader(ds, 32, workers=a.workers, pin=dev.type == "cuda",
                         collate_fn=passthrough if a.batched else None)
    model = make_model().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    sched = schedule(wait=1, warmup=1, active=4)          # skip 1 step, warm up 1, record 4
    it = iter(loader)
    with profile(activities=activities, schedule=sched) as prof:
        for _ in range(a.steps):
            with record_function("data_wait"):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(loader)
                    x, y = next(it)
            with record_function("train_step"):
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(set_to_none=True)
                loss_fn(model(x), y).backward()
                opt.step()
            sync()
            prof.step()                                    # tell the schedule a step ended

    sort = "cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort, row_limit=8))
    prof.export_chrome_trace(a.out)
    print(f"trace written to {a.out}: open it at https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
