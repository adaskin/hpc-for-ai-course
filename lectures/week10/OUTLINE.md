# Week 10 outline: Feeding the GPU (data pipelines for deep learning)

*YZM 513, Fall 2026. Outline only: the Marp deck, scripts and lab come next, in the Week 6 / Week 9 style.*

**Assumption.** The README schedule is one week behind the course as taught, so the next lecture is **Week 10 = README "Week 9: Data Pipelines for Deep Learning"**. The Week 9 deck also ends with "Next week: feeding the GPU". Project 2 proposals are due this week, so the session reserves a slot for them.

## Approach: follow the official PyTorch docs

The deck is anchored on one official tutorial, so students can re-read the same material afterwards. Its progression (baseline, workers, `pin_memory`, persistent workers, prefetcher, `__getitems__`) is the spine of Part II.

| Part | Official page | What to lift |
|---|---|---|
| I, II | [Data Loading Optimization in PyTorch](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html) | synthetic dataset, baseline, step-by-step cumulative speedups, shared-memory section |
| II, IV | [`torch.utils.data` / `DataLoader`](https://docs.pytorch.org/docs/stable/data.html) | knobs and warnings (`in_order`, start method, seeding, memory) |
| III | [Good usage of `non_blocking` and `pin_memory()`](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) | what overlaps and what does not, the two correctness hazards |
| V | [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | `schedule`, `record_function`, Chrome-trace export |
| V, VI | [Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | avoiding syncs, `DataLoader` settings, `set_to_none`, `cudnn.benchmark` |

To keep it understandable, the deck stays short on the *core* ladder and treats everything else as brief "know it exists" slides, as in Week 9.

## The week in one sentence

> A GPU that waits for its input is a GPU you paid for and did not use. Week 9 made the kernels fast; Week 10 makes sure they never have to wait.

## Learning goals

By the end of the session a student can:

1. **Diagnose** whether a training job is input-bound, from a profiler trace and from timing the data wait separately from the step, and not only from `nvidia-smi`.
2. **Apply the ladder** (workers, pinned memory, persistent workers, prefetching, batched `__getitems__`), predict each step's effect, and say which resource it spends (CPU cores, RAM, PCIe).
3. **Explain** what actually overlaps copies with compute, and why `pin_memory=True` plus `non_blocking=True` alone does not guarantee it.
4. **Avoid the hazards:** hidden synchronisation points that make a job look input-bound, and unsafe `non_blocking` copies.
5. **Fix** a deliberately slow script and report cumulative speedup with evidence, including steps that gave no gain.

## Where it sits (review slide)

| Earlier week | Returns today as |
|---|---|
| 1: profiling, Amdahl | the data stage is the serial part; speeding up the GPU alone hits Amdahl's wall |
| 2: vectorisation | `__getitems__`: one batched call instead of many per-sample calls |
| 3-4: GIL, processes, worker pools, races | `DataLoader` workers are processes for exactly this reason; the queue is a producer/consumer pool; `non_blocking` misuse is a race on PCIe |
| 5: the four questions (split, assign, move, wait); stragglers | split = batches, assign = workers, move = copy, wait = prefetch queue; `in_order=False` and slow workers |
| 6: transfers | host-to-device copy cost |
| 9: streams, pinned memory, timeline | the three requirements for copy/compute overlap; "gaps between kernels" is today's diagnosis |

## Deck parts

**I. The idle GPU (core).** Open with a profiler timeline of a slow script, with visible gaps (bridge from the Week 9 closing question). Cost model: serial step = t_data + t_gpu; fully overlapped step = max(t_data, t_gpu). *Worked example:* 30 ms + 50 ms = 80 ms (GPU 62.5% busy) vs 50 ms (100%).

**II. The optimisation ladder (core).** One shared training loop and the official synthetic dataset (512 samples, batch 32, 5 ms of simulated per-sample transform work). For each step: **predict, run, record the cumulative speedup**.
1. Baseline: `num_workers=0`, `pin_memory=False` (the loader runs in the main process).
2. `num_workers` (and `prefetch_factor`, default 2 batches per worker).
3. `pin_memory=True`.
4. `persistent_workers=True`.
5. Overlapping the host-to-device copy with compute (the tutorial's `DataPrefetcher`).
6. `__getitems__`: the `DataLoader` calls it once with the whole list of indices if the dataset defines it, so the per-sample work can be vectorised (Week 2 returns).

*What the tutorial reports (partial read of the page; hardware not stated; re-run on the T4 before quoting):* workers about 3.4x, `pin_memory` about 1.0x (no gain, because CPU transform time dominates), persistent workers about 1.1x, prefetcher about 1.0x extra, `__getitems__` about 4x extra (roughly 15x in total). Teach the null results on purpose: a knob only helps if it removes the actual bottleneck.

*Worker sizing, one slide:* workers needed ~ GPU consumption rate / per-worker delivery rate (e.g. 2,560 samples/s needed at batch 128 and 50 ms/step; a worker delivering 100/s means 26). Past the core count more workers stop helping; check `os.cpu_count()` (the `DataLoader` itself warns when `num_workers` exceeds what it suggests). The lesson is "when workers run out, reduce per-sample work", which is the `__getitems__` / caching step, so the lab does not depend on core count.

**III. Getting bytes to the GPU, correctly (core).**
- Pageable vs pinned host memory; `to(device, non_blocking=True)` returns control to the host early when the source is pinned.
- **Copy/compute overlap on the device needs all three:** a free DMA engine, a **separate non-default CUDA stream**, and a pinned source (Week 9, Part VIII). `pin_memory=True` plus `non_blocking=True` on the default stream does not by itself give overlap; the prefetcher with its own stream does.
- Calling `tensor.pin_memory().to(device, non_blocking=True)` by hand can be up to twice as slow as plain `.to(device)`, because `pin_memory()` blocks the host. Let the `DataLoader` pin batches (`pin_memory=True`) instead. The tutorial's single-tensor numbers: 0.375 ms pageable `.to()`, 0.428 ms for `pin_memory().to()`.
- **Hazard slide (a Week 3 race on PCIe):** mutating a pinned source after a `non_blocking` host-to-device copy can corrupt what the GPU receives; `non_blocking` device-to-host without `torch.cuda.synchronize()` can give garbage. The tutorial's tests fail at iteration 2 and 0 respectively. Demo it, then fix it with a sync or event.
- Bytes still count: a 128 x 3 x 224 x 224 float32 batch is 77 MB, and as uint8 it is 19 MB (ship uint8, normalise on the GPU). Copy time to be measured on the T4, not assumed.

**IV. Worker pitfalls (brief, 2 slides).** One slide each, pointing at the `DataLoader` docs:
- Start method and workers: Python 3.14 changed the default start method on non-Mac POSIX to `forkserver`, so the dataset and `worker_init_fn` must be picklable (no lambdas); guard with `if __name__ == "__main__"`. Check the Python version on Colab.
- Seeding and memory: random augmentations must differ across workers (read the docs' randomness notes); Python-object datasets can grow memory per worker because of refcounting and copy-on-write, so prefer NumPy arrays.
- `in_order=False` lets a slow worker not block the queue (a Week 5 straggler). The docs warn it can harm reproducibility and skew the data distribution on imbalanced data; use it knowingly.
- Shared memory: `file_descriptor` vs `file_system` sharing strategies, `set_sharing_strategy()` before creating loaders with workers, and the "bus error" from a full `/dev/shm` on Docker/Colab.

**V. Measuring it right (core).**
- `nvidia-smi` utilisation only says some kernel was running; it is a hint, not a diagnosis.
- **Hidden syncs:** `.item()`, `.cpu()`, `print(cuda_tensor)`, `nonzero()` and Python control flow on CUDA values force the CPU to wait for the GPU, so the job can look input-bound when it is sync-bound. The tutorial's own loop calls `loss.item()` every batch: show what it costs and how to log every N steps instead.
- **Timing recipe:** separate host-only data wait from the step.
  ```python
  torch.cuda.synchronize(); t0 = time.perf_counter()
  batch = next(it)
  t1 = time.perf_counter()              # data wait: host time, no sync before this stop
  ...  # forward, backward, optimizer step
  torch.cuda.synchronize(); t2 = time.perf_counter()   # step time
  ```
  Do not sync before `t1`: that would add the previous step's GPU work to the data wait. The sync before `t0` serialises CPU and GPU a little, so the trace (below) is the unperturbed view.
- `torch.profiler` with `schedule(wait, warmup, active)`, `record_function("data_wait")` labels and `export_chrome_trace`; open the JSON in Perfetto or `chrome://tracing`, as in Week 9. The profiler recipe I read shows the Chrome trace viewer; I did not see the TensorBoard plugin mentioned there, so the deck uses Perfetto/Chrome unless the plugin is re-checked.

**VI. Tuning recipe (core, one slide + one table).** Measure first; then apply the ladder in order and stop when the loader is no longer the bottleneck; remove unnecessary syncs. From the tuning guide: `optimizer.zero_grad(set_to_none=True)`, `torch.backends.cudnn.benchmark = True` for fixed-size convolutional inputs (autotuning can be non-deterministic). Include the honest case: "the job is not input-bound, so none of this helps".

**VII. Where the data lives (brief, 1 slide).** Not covered by the official tutorial, so only a pointer: many small files vs sharded streams (WebDataset), Parquet/Arrow, `np.memmap`; decoded-and-cached datasets; GPU-side decode and augmentation (DALI). Point to each project's own docs.

**VIII. Project 2 proposals (10 min).** Walk through the one-page proposal checklist from the brief (system, workload, hardware, baseline, question, at least two interventions, metrics). Track A students say which pipeline measurements they will take.

**IX. Lab and wrap-up.** Lab below, takeaways, forward map, "Before next week", references.

## Planned scripts (`lectures/week10/`)

| File | Part | What it shows |
|---|---|---|
| `01_cost_model.py` | I | serial vs overlapped step time and worker-count arithmetic; **no GPU needed** |
| `02_ladder.py` | II | the six steps on the synthetic dataset; prints a cumulative speedup table |
| `03_pinned_transfer.py` | III | pageable vs pinned, blocking vs `non_blocking`, manual `pin_memory()`; the two hazard demos with a sync fix |
| `04_hidden_syncs.py` | V | the same loop with `loss.item()` every step vs every N steps; data-wait vs step timing with the corrected recipe |
| `05_profile_pipeline.py` | V | `torch.profiler` trace with `data_wait` labels, exported for Perfetto |
| `06_real_images.py` | lab stage 2 | optional: CIFAR-10 or generated JPEGs, decode and augment cost, cached decoded data, uint8 transfer |
| `gpu_utils.py`, `make_figures.py` | all | timing helpers; regenerates the `img/w10-*.svg` figures |

Every step's output is checked against the baseline loop (same loss for the same data) before it is timed.

## Lab: "Your GPU is idle. Find out why and fix it"

**Stage 1: synthetic dataset (works on any core count).**
1. Run the baseline; record seconds per epoch, data-wait share and a profiler trace.
2. **Predict** each step's speedup, apply the ladder one step at a time, and fill in the cumulative speedup table. Idealised arithmetic gives a first prediction: the per-sample delay cost is `samples x 5 ms`, divided by the number of workers at best.
3. Add the corrected data-wait timing and `record_function` labels; compare the trace before and after.
4. **Deliverable:** the table plus the sentence *"my job was ______-bound, because ______ ms of each ______ ms step was spent waiting for data; the step that helped most was ______, and ______ gave no gain because ______."*

**Stage 2 (optional, real images).** Decode, resize and augment cost, cached decoded data and uint8 transfer; before/after images/s, with the loss curve unchanged. Expect decoding to dominate; measure the gain, do not assume it.

## Takeaways (draft)

1. Measure the wait before touching any knob; the knobs only help an input-bound job.
2. Climb the ladder in order and re-measure; expect some steps to give nothing.
3. The biggest lever is usually less per-sample work (batched, cached, smaller dtype), not more workers.
4. Overlap needs a pinned source, a separate stream and a free DMA engine; `non_blocking` without a sync is a race.
5. Hidden syncs make an idle GPU look like a slow loader; use the trace, not just `nvidia-smi`.

## Forward map

| today | where it reappears |
|---|---|
| input throughput and `DistributedSampler` | per-GPU data feeding in DDP (Weeks 11-12) |
| copy/compute overlap, hidden syncs | DDP communication overlap (Weeks 11-12), mixed-precision and `torch.compile` benchmarks (Weeks 12-13) |
| the measurement recipe | Project 2 reports (all tracks) |

## Before next week

- Finish the lab notebook; one paragraph on what the bottleneck was and which step finally removed it.
- Submit the Project 2 proposal.
- Read: the official data-loading tutorial (the steps you did not reach) and the `DataLoader` docs warnings.
- Think ahead: with 2 GPUs, how does each one get different data? (Week 11.)

## References

- PyTorch, *Data Loading Optimization in PyTorch*; *`torch.utils.data`*; *A guide on good usage of `non_blocking` and `pin_memory()`*; *PyTorch Profiler* recipe; *Performance Tuning Guide* (links in the table above).
- WebDataset, Apache Arrow / Parquet, NVIDIA DALI documentation (pointers only).
- Course repo: Week 9 deck, Part VIII (streams) and the profiling section.

## What I checked (before building the deck)

| Claim | Status |
|---|---|
| The official tutorial exists and has the ladder above, the shared-memory section and a `__getitems__` step | read the page (the end of the page came as a summary) |
| Its reported speedups (about 3.4x workers, 1.0x pin, 1.1x persistent, 1.0x prefetcher, about 4x `__getitems__`) | from that read; hardware not stated; its own summary table may differ, so re-run before quoting |
| Three overlap requirements; manual `pin_memory()` up to 2x slower; the two hazards | pinmem page (hazards and numbers read directly; the requirements came from a summary) |
| `in_order` and its warning; `forkserver` on Python 3.14+; `prefetch_factor` and `persistent_workers` semantics; `__getitems__` hook | PyTorch source docstrings on `main`; check they match the PyTorch version on Colab |
| Hidden-sync list, `set_to_none`, `cudnn.benchmark` | tuning guide (partial read); re-read before teaching |
| An earlier suggestion that the lab would run baseline about 25 s, workers about 13 s, `__getitems__` about 1 s (32x), and that caching gives 5-10x | **not from any page**; the official page reports about 4x for `__getitems__`. Treat as untested predictions, measure on the T4 |
| Worker seeding and memory-growth warnings | pointer to the docs' randomness notes; not read |
| TensorBoard profiler plugin status | not mentioned on the recipe page I read; deck uses Perfetto |

Nothing here has been run (no GPU in my environment).

## Open items

- **Confirm the week number** (assumed Week 10). A schedule footer like Week 9's would cover it, or the README can be renumbered in a separate PR.
- Run the official tutorial on a Colab T4 first, and record the Python and PyTorch versions and `os.cpu_count()`.
