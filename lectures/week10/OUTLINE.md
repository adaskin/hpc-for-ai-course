# Week 10 outline: Feeding the GPU (data pipelines for deep learning)

*YZM 513, Fall 2026. Outline only: the Marp deck, scripts and lab come next, in the Week 6 / Week 9 style.*

**Assumption.** The README schedule is one week behind the course as taught (the midterm review became the midterm, so Weeks 8 and 9 are the kernel weeks). The next lecture is therefore **Week 10 = README "Week 9: Data Pipelines for Deep Learning"**, which is also where the Week 9 deck points ("Next week: feeding the GPU"). Project 2 proposals are due this week, so the session reserves a slot for them.

## The week in one sentence

> A GPU that waits for its input is a GPU you paid for and did not use. Week 9 made the kernels fast; Week 10 makes sure they never have to wait.

## Learning goals

By the end of the session a student can:

1. **Diagnose** whether a training job is input-bound, from a profiler trace and from three numbers (time waiting for data, time computing, GPU utilisation), and not only from `nvidia-smi`.
2. **Explain** what each `DataLoader` knob does and which hardware resource it spends: `num_workers` (CPU cores), `pin_memory` and `non_blocking` (PCIe), `prefetch_factor` and `persistent_workers` (host RAM, start-up cost).
3. **Estimate** how many workers a job needs from the GPU step time and the per-sample CPU cost, and say when more workers cannot help (too few cores, slow storage).
4. **Choose** a storage layout (many small files vs sharded streams vs memory-mapped arrays) and say why sequential reads win.
5. **Fix** a deliberately slow training script and report before/after throughput with evidence.

## Where it sits (review slide, one line per earlier week)

| Earlier week | Returns today as |
|---|---|
| 1: profiling, Amdahl | the data stage is the serial part; speeding up the GPU alone hits Amdahl's wall |
| 3-4: GIL, processes, worker pools | `DataLoader` workers are processes for exactly this reason; the queue between them is a producer/consumer pool |
| 5: the four questions (split, assign, move, wait) | split = batches, assign = workers, move = pinned copy, wait = prefetch queue |
| 6: transfers, arithmetic intensity | host-to-device copy cost; a transfer is a "kernel" with intensity 0 |
| 9: streams, pinned memory, profiler timeline | `pin_memory` + `non_blocking=True` is the same overlap; "gaps between kernels" is today's diagnosis |

## Deck parts

Core parts are taught in full; brief parts are "know what it is", as in Week 9.

**I. The idle GPU (core).** Open with a profiler timeline of a slow script, with visible gaps. The cost model: serial step = t_data + t_gpu, overlapped step = max(t_data, t_gpu). *Worked example:* t_data = 30 ms, t_gpu = 50 ms gives 80 ms (GPU 62.5% busy) vs 50 ms (100%). Bridge from Week 9's closing question.

**II. Anatomy of a `DataLoader` (core).** Dataset (map-style vs iterable), sampler, batch sampler, `collate_fn`, worker processes, the prefetch queue, the pin-memory thread. One diagram of the whole path from disk to GPU.
- *Example 1: the knobs.* `num_workers` 0 vs N, with the "why does 0 mean the main process does everything" explanation.
- *Example 2: sizing workers.* GPU step 50 ms for batch 128 needs 2,560 samples/s; one worker at 10 ms per sample delivers 100/s, so >= 26 workers. On a 2-core Colab VM the ceiling is about 200/s, so step time becomes ~640 ms and the GPU is ~8% busy (a 50,000-sample epoch: ~250 s instead of ~20 s). *Illustrative numbers; replace with measured ones from the Colab run.*
- Worker pitfalls: `if __name__ == "__main__"` and spawn vs fork, copy-on-write and per-worker memory growth with Python-object datasets (prefer NumPy arrays or memory maps), checking that random augmentations differ across workers, `IterableDataset` needs explicit sharding with `get_worker_info()`.

**III. Getting bytes to the GPU (core).** Pageable vs pinned memory, `pin_memory=True`, `tensor.to(device, non_blocking=True)`, why it only overlaps with pinned memory (Week 9, streams). *Worked example:* a 128 x 3 x 224 x 224 float32 batch is 77 MB; the same batch as uint8 is 19 MB. Ship uint8, normalise on the GPU: 4x fewer bytes over PCIe (copy time to be measured on the T4, not assumed).

**IV. Where the data lives (brief).** The small-files problem; sequential vs random reads; sharded tar streams (WebDataset), TFRecord, Parquet/Arrow, `np.memmap`; decode cost (JPEG) vs storage cost; caching a decoded, resized dataset. GPU-side decode and augmentation (NVIDIA DALI, torchvision transforms on GPU tensors) as a one-slide pointer.

**V. Measuring it right (core).** `nvidia-smi` utilisation only says a kernel was running, not that the GPU was well used. Three better signals: (1) time in `next(loader)` vs time in the step, with a `cuda.synchronize()` before each clock stop; (2) `torch.profiler` with `schedule(wait, warmup, active)` and `record_function("data_wait")` labels; (3) images/s. View the trace in Perfetto (as in Week 9). TensorBoard's profiler plugin is in the README, but check that it is still maintained and installs on Colab before class; fall back to Perfetto if not.

**VI. A tuning recipe (core).** Order matters: measure, then workers, then pinned memory, then prefetch and persistent workers, then storage format, then GPU-side transforms. Stop when the loader is no longer the bottleneck; more is waste (RAM, start-up time). Includes the honest case: "the job is not input-bound, so none of this helps".

**VII. Project 2 proposals (10 min).** Walk through the one-page proposal checklist from the brief: system, workload, hardware, baseline, question, at least two interventions, metrics. Track A students should say which pipeline measurements they will take.

**VIII. Lab and wrap-up.** Lab below, takeaways, forward map, "Before next week", references.

## Planned scripts (`lectures/week10/`, every loader checked against a baseline before timing)

| File | Part | What it shows |
|---|---|---|
| `01_cost_model.py` | I | serial vs overlapped step time and worker-count arithmetic; **no GPU needed** |
| `02_slow_training.py` | I, V | the provided script: per-sample file reads and PIL transforms, `num_workers=0`, small batch; prints images/s, data-wait share, GPU busy share |
| `03_loader_sweep.py` | II | sweep `num_workers` {0, 1, 2, 4}, `prefetch_factor`, `persistent_workers`; results table |
| `04_pinned_transfer.py` | III | pageable vs pinned, blocking vs `non_blocking`, float32 vs uint8 transfer; GB/s |
| `05_storage_formats.py` | IV | many files vs one shard vs `np.memmap`, sequential vs shuffled reads |
| `06_profile_pipeline.py` | V | `torch.profiler` trace with `data_wait` labels, exported for Perfetto |
| `gpu_utils.py`, `make_figures.py` | all | shared timing helpers; regenerates the `img/w10-*.svg` figures |

## Lab: "Your GPU is idle 60% of the epoch. Find out why and fix it"

1. Run `02_slow_training.py`; record images/s, data-wait share and GPU utilisation.
2. **Predict** the bottleneck, then open the profiler trace and confirm or correct the prediction.
3. Apply one change at a time (workers, pinned memory + `non_blocking`, prefetch, persistent workers, cached or sharded data, uint8 transfer); re-measure after each.
4. Check that the model's loss curve and final accuracy did not change (correctness first, as in Weeks 8 and 9).
5. **Deliverable:** a results table and the sentence *"my job was ______-bound, because ______ ms per step waiting for data out of ______ ms, and after the fix ______ images/s."*

Colab note: the free VM has only a few CPU cores (check `os.cpu_count()`), so the lab is partly about learning when workers stop helping. Report that result honestly.

## Takeaways (draft)

1. Measure the wait before touching any knob; the knobs only help an input-bound job.
2. Workers spend CPU cores; pinned memory spends RAM and enables overlap; prefetch spends RAM; each has a limit.
3. Bytes moved over PCIe count, just like bytes moved from VRAM in Week 9: ship small types, convert on the GPU.
4. Sequential reads from a few large shards beat random reads from many small files.
5. `nvidia-smi` utilisation is a hint, not a diagnosis; use the trace.

## Forward map

| today | where it reappears |
|---|---|
| input pipeline throughput | per-GPU data feeding in DDP, `DistributedSampler` (Weeks 11-12) |
| pinned memory, overlap | DDP communication overlap (Weeks 11-12) |
| profiler traces | mixed precision and `torch.compile` benchmarks (Weeks 12-13) |
| storage formats, caching | Project 2, Track A (data loading and transfer overlap) |

## Before next week

- Finish the lab notebook; one paragraph on what the bottleneck was and what finally removed it.
- Submit the Project 2 proposal.
- Read: PyTorch `DataLoader` docs (the `num_workers`, `pin_memory`, `prefetch_factor` sections) and the PyTorch profiler recipe.
- Think ahead: with 2 GPUs, how does each one get different data? (Week 11.)

## Readings and references

- PyTorch docs: `torch.utils.data` (`DataLoader`, `IterableDataset`, `get_worker_info`), the "pin_memory and non_blocking" tutorial, the profiler recipe.
- WebDataset and Apache Arrow / Parquet documentation (brief).
- NVIDIA DALI documentation (pointer only).
- Course repo: Week 9 deck, Part VIII (streams) and the profiling section.

## Open items to settle before building the deck

- **Confirm the week number** (assumed Week 10). The README schedule still has the old numbering; a one-line schedule footer like Week 9's would do, or the README can be renumbered in a separate PR.
- **Dataset for the lab.** Suggest CIFAR-10 or a generated image folder; the point is a realistic CPU cost per sample, not accuracy.
- **Verify before teaching:** TensorBoard profiler plugin status on Colab; the Colab VM's core count and whether `prefetch_factor` and `persistent_workers` behave as described on the installed PyTorch version. Nothing in this outline has been run yet (no GPU in my environment).
