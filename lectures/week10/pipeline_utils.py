"""Shared pieces for the Week 10 scripts.

The synthetic dataset and the "ladder" of optimisations follow PyTorch's official
tutorial *Data Loading Optimization in PyTorch*
(https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html).
This is our own small implementation, so the numbers it prints are ours, not the tutorial's.

Contents
    SyntheticDataset / BatchedDataset   fake data with a fixed cost per call
    passthrough                         collate function for pre-batched data
    make_loader                         DataLoader with the knobs we sweep
    CudaPrefetcher                      overlaps the host-to-device copy with compute (CUDA only)
    run_epochs                          a tiny training loop that times data wait and step time
"""
import time

import numpy as np
import torch
from torch import nn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    """Wait for the GPU. A no-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- data
class SyntheticDataset(torch.utils.data.Dataset):
    """512 feature vectors by default. Every __getitem__ call costs `delay` seconds.

    The delay stands for the fixed cost of one call: opening a file, calling a decoder.
    The transform (normalise to zero mean, unit variance) is real work on the sample.
    """

    def __init__(self, size=512, feature_dim=224, delay=0.005, seed=0):
        rng = np.random.default_rng(seed)
        self.x = rng.standard_normal((size, feature_dim)).astype(np.float32)
        self.y = rng.integers(0, 10, size).astype(np.int64)
        self.delay = delay

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        time.sleep(self.delay)                       # fixed cost of one call
        x = self.x[i]
        x = (x - x.mean()) / (x.std() + 1e-6)        # per-sample transform
        return torch.from_numpy(x), int(self.y[i])


class BatchedDataset(SyntheticDataset):
    """Same data, but the DataLoader can ask for a whole batch in one call.

    If a dataset defines __getitems__, the DataLoader calls it once with the list of
    indices of the batch instead of calling __getitem__ once per index.
    """

    def __getitems__(self, indices):
        time.sleep(self.delay)                       # the fixed cost is paid once per batch
        idx = np.asarray(indices)
        x = self.x[idx]
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


def passthrough(batch):
    """collate_fn for BatchedDataset: the batch is already a pair of tensors.

    Defined at module level (not a lambda) so worker processes can pickle it.
    """
    return batch


def make_loader(dataset, batch_size=32, workers=0, pin=False, persistent=False,
                prefetch_factor=None, collate_fn=None):
    kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=workers,
                  pin_memory=pin, collate_fn=collate_fn)
    if workers > 0:                                  # these two are only valid with workers
        kwargs["persistent_workers"] = persistent
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(dataset, **kwargs)


# ------------------------------------------------------------------ copy/compute overlap
class CudaPrefetcher:
    """Copy batch k+1 to the GPU on a side stream while batch k is being computed.

    All three overlap requirements are visible here: a pinned source (the loader's
    pin_memory=True), a separate non-default stream, and non_blocking=True copies.
    """

    def __init__(self, loader, device):
        self.it = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self._preload()

    def _preload(self):
        try:
            x, y = next(self.it)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = (x.to(self.device, non_blocking=True),
                          y.to(self.device, non_blocking=True))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)     # copy finished?
        batch = self.batch
        for t in batch:                                          # tell the allocator who uses it
            t.record_stream(torch.cuda.current_stream())
        self._preload()
        return batch


# ------------------------------------------------------------------------------ training
def make_model(feature_dim=224):
    return nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, 10))


def run_epochs(loader, epochs=3, non_blocking=False, prefetch=False, item_every=1, device=None):
    """Train a tiny model for `epochs` passes over `loader`.

    Timing follows the recipe from the slides:
      data wait = host time inside next(it), no sync before it is stopped
      step time = forward + backward + update, with a sync before the clock stops
    Returns a dict with seconds, data_wait, step, steps, checksum (sum of all input
    values, used to prove that every configuration sees the same data) and loss.
    """
    device = device or get_device()
    torch.manual_seed(0)
    model = make_model().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    data_wait = step_time = 0.0
    steps = 0
    checksum = torch.zeros((), dtype=torch.float64, device=device)
    total_loss, loss = 0.0, None

    sync()
    start = time.perf_counter()
    for _ in range(epochs):
        it = iter(CudaPrefetcher(loader, device) if prefetch else loader)
        while True:
            sync(); t0 = time.perf_counter()
            try:
                x, y = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()                 # data wait: host time, no sync before this
            if not prefetch:
                x = x.to(device, non_blocking=non_blocking)
                y = y.to(device, non_blocking=non_blocking)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            if steps % item_every == 0:
                total_loss += loss.item()            # .item() waits for the GPU
            checksum += x.double().sum()
            sync(); t2 = time.perf_counter()
            data_wait += t1 - t0
            step_time += t2 - t1
            steps += 1
    sync()
    return dict(seconds=time.perf_counter() - start, data_wait=data_wait, step=step_time,
                steps=steps, checksum=checksum.item(), loss=float(loss.item()))
