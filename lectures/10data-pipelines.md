---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 28px;
  }
  section.lead { text-align: center; }
  section.lead h1 { font-size: 52px; }
  table { font-size: 24px; }
  code { font-size: 22px; }
  pre { font-size: 19px; }
  img:not(.emoji) { display: block; margin: 0 auto; }   /* emoji images stay inline */
  footer { font-size: 16px; opacity: 0.5; }
  /* html helpers: columns, cards, tags (render with html enabled) */
  .cols  { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: start; }
  .cols3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; align-items: start; }
  .wide-left { display: grid; grid-template-columns: 1.5fr 1fr; gap: 22px; align-items: start; }
  .card { border: 2px solid #cbd5e1; border-radius: 14px; padding: 6px 16px; background: #f8fafc; font-size: 24px; }
  .card h3 { margin: 4px 0; font-size: 28px; }
  .card p { margin: 4px 0; }
  .blue   { border-color: #2f6fb0; background: #e6eef8; }
  .orange { border-color: #e08a1e; background: #fbeedb; }
  .green  { border-color: #2e7d5b; background: #e6f2ec; }
  .red    { border-color: #b3402f; background: #f7e4e0; }
  .gray   { border-color: #6b7280; background: #eef0f3; }
  .big    { font-size: 36px; }
  .small  { font-size: 21px; }
  .tag { display: inline-block; padding: 0 12px; border-radius: 12px; background: #2f6fb0; color: white; font-size: 20px; }
  .tag.orange { background: #e08a1e; border: 0; }
  .tag.green  { background: #2e7d5b; border: 0; }
  .tag.red    { background: #b3402f; border: 0; }
---

<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _footer: "Schedule note: original syllabus Week 9 topic, taught in Week 10." -->

# YZM 513 – High-Performance Programming for AI

### Week 10: Feeding the GPU 🚰
### DataLoader pipelines · pinned memory · measuring the wait

**İstanbul Medeniyet Üniversitesi**
AI Engineering M.Sc. – Fall 2026

Ammar Daşkın

---

## 🗺️ Where we are

```text
Week 1–2    one core, made fast           vectorise, Numba, BLAS
Week 3–4    one machine, many cores       threads, processes, async, pools
Week 5      many machines                 Dask, Ray, cost of distribution
Week 6      one GPU, thousands of cores   architecture + CuPy
Week 7      midterm
Week 8      writing kernels               indexing, bounds, launches, transfers
Week 9      making kernels FAST           memory hierarchy, tiling, profiling
Week 10     feeding the GPU               DataLoader pipelines     ← TODAY
Week 11–12  many GPUs                     DDP, model parallelism
Week 13     deployment                    torch.compile
```

> 🚰 A GPU that waits for its input is a GPU you paid for and did not use.

---

## 🔁 Last week in one minute

<div class="cols">
<div class="card blue">

### 📏 Count the bytes
A kernel is fast when each byte from VRAM is **used many times**.
<span class="small">➡️ Today: bytes also cross the slower PCIe link (~12 GB/s).</span>

</div>
<div class="card orange">

### 🧱 Neighbours together
Coalescing and tiling: one big aligned fetch beats many scattered ones.
<span class="small">➡️ Today: one batched call beats many per-sample calls.</span>

</div>
<div class="card green">

### 🌊 Overlap
Streams and pinned memory hide copies behind compute.
<span class="small">➡️ Today: the same trick, for the input batch.</span>

</div>
<div class="card red">

### 📈 The timeline
Gaps between kernels mean an **idle GPU**.
<span class="small">➡️ Today: why are there gaps, and how do we close them?</span>

</div>
</div>

---

## 🧠 PyTorch in one slide (if you are new to it)

<div class="wide-left">
<div>

```python
loader = DataLoader(dataset, batch_size=32)
model = nn.Linear(224, 10).to("cuda")

for x, y in loader:            # one pass = one epoch
    x, y = x.to("cuda"), y.to("cuda")
    loss = loss_fn(model(x), y)  # forward
    loss.backward()              # gradients
    optimizer.step()             # update weights
    optimizer.zero_grad()
```

</div>
<div class="card gray">

**sample**: one example

**batch**: a group of samples

**step**: one loop pass, one batch

**epoch**: every sample once

The GPU cannot start a step until **its batch has arrived**.

</div>
</div>

---

<!-- _class: lead -->

# ⏱️ The idle GPU

---

![w:900](img/w10-idle-gpu.svg)

<!-- _footer: "Schematic: block widths are illustrative (30 ms of data work, 50 ms of GPU work)." -->

---

## Cost model: who sets the pace?

| | step time | GPU busy |
|---|---:|---:|
| **serial** (load, then compute) | 30 + 50 = **80 ms** | 62 % |
| **overlapped** (load while computing) | max(30, 50) = **50 ms** | 100 % |

If the loader needs **more** than the GPU (say 80 ms vs 50 ms), the job is **input-bound**: a faster GPU changes nothing.

🎯 Two jobs for today:
- **Overlap** loading with compute
- **Shrink** the loading work itself

<!-- _footer: "Try it: python week10/01_cost_model.py --t-data 80 --t-gpu 50" -->

---

## 🧰 Anatomy of a DataLoader

![w:940](img/w10-dataloader-path.svg)

- Workers are **processes** (Weeks 3–4: the GIL) building batches in parallel
- The main loop only **takes** ready batches, so it waits when the queue is empty

---

<!-- _class: lead -->

# 🪜 The optimisation ladder

### One change at a time, following PyTorch's official tutorial

---

## 🧪 Our test bench

Modelled on the official tutorial *Data Loading Optimization in PyTorch*: **512 samples, batch 32, a 5 ms fixed cost per call** (opening a file, calling a decoder).

```python
class SyntheticDataset(Dataset):
    def __getitem__(self, i):
        time.sleep(self.delay)                  # fixed cost of one call
        x = self.x[i]
        x = (x - x.mean()) / (x.std() + 1e-6)   # real per-sample work
        return torch.from_numpy(x), int(self.y[i])
```

Same training loop for every step. 🔒 **Correctness first:** the script checks every step saw **the same data** as the baseline.

<!-- _footer: "Full code: week10/pipeline_utils.py (dataset, loop, timing) · week10/02_ladder.py (the whole ladder)" -->

---

## 🐢 Baseline: `num_workers=0`

```python
loader = DataLoader(dataset, batch_size=32)      # the default
```

The **main process** does everything, one after the other:

> load batch → train on it → load the next batch → …

- Each epoch: 512 samples × 5 ms ≈ **2.6 s of pure waiting**
- The profiler would show the GPU idle most of the time
- This is the number every other step is compared with

<!-- _footer: "The default num_workers=0 means loading is synchronous and done in the main process (PyTorch Performance Tuning Guide)." -->

---

## 🧵 Workers: `num_workers`

```python
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

<div class="cols">
<div>

- Each worker is a **separate process** that builds whole batches
- `prefetch_factor` (default 2): batches each worker prepares **ahead of time**
- Different processes, so **no GIL problem** (Weeks 3–4)

</div>
<div class="card orange">

**Costs:** CPU cores, RAM, start-up time.

**How many?** About *(samples/s the GPU eats) ÷ (samples/s one worker makes)*. More workers than **cores** will not help: check `os.cpu_count()`.

</div>
</div>

<!-- _footer: "Worker sizing: week10/01_cost_model.py" -->

---

## 📌 Pinned memory: `pin_memory=True`

```python
loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

- **Pageable** RAM can be moved by the OS, so the GPU cannot read it directly: the driver stages a copy first.
- **Pinned** (page-locked) RAM stays put: the GPU's copy engine (**DMA**) reads it straight away.
- The DataLoader pins each batch in a **separate thread**.

⚠️ Pinning does not make the loader faster. It makes the **copy** faster. When CPU work dominates, expect **no gain**. That is a normal result.

---

## ♻️ Persistent workers: `persistent_workers=True`

```python
loader = DataLoader(dataset, batch_size=32, num_workers=4,
                    pin_memory=True, persistent_workers=True)
```

- By default workers are **shut down after every epoch** and started again
- `persistent_workers=True` keeps them (and their dataset copies) alive
- 👍 Saves start-up time. Matters most with **many short epochs**
- 👎 Keeps the workers' memory in use between epochs

---

## 🌊 Copy on a side stream (prefetcher)

![w:860](img/w10-overlap.svg)

<!-- _footer: "Week 9 streams, applied to the input batch. Official guide: A guide on good usage of non_blocking and pin_memory()." -->

---

## 🌊 The prefetcher: the important parts

```python
class CudaPrefetcher:
    def __init__(self, loader, device):
        self.it, self.device = iter(loader), device
        self.stream = torch.cuda.Stream()              # the side stream
        self._preload()

    def _preload(self):                                # copy batch k+1 ...
        x, y = next(self.it)
        with torch.cuda.stream(self.stream):
            self.batch = (x.to(self.device, non_blocking=True),
                          y.to(self.device, non_blocking=True))

    def __next__(self):                                # ... while batch k computes
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self._preload()
        return batch
```

<!-- _footer: "Full version (end of data, record_stream): week10/pipeline_utils.py" -->

---

## 📦 `__getitems__`: ask for the whole batch

By default the DataLoader calls `dataset[i]` **once per sample**. If your dataset defines `__getitems__`, it is called **once per batch** with the list of indices.

```python
class BatchedDataset(SyntheticDataset):
    def __getitems__(self, indices):
        time.sleep(self.delay)                   # fixed cost: once per batch
        x = self.x[np.asarray(indices)]          # one fancy-indexing read
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        return torch.from_numpy(x), torch.from_numpy(self.y[indices])

loader = DataLoader(ds, batch_size=32, num_workers=4, collate_fn=passthrough)
```

🔁 **Week 2 again:** vectorise. One big operation beats 32 small ones.

<!-- _footer: "collate_fn=passthrough because the batch is already built (a module-level function, so workers can pickle it). Full code: week10/pipeline_utils.py" -->

---

## 📊 The ladder at a glance

| Rung | Spends | Helps when | Tutorial reports |
|---|---|---|---:|
| 🧵 workers | CPU cores, RAM | loading is slow | ≈ 3.4× |
| 📌 `pin_memory` | pinned RAM | the copy is slow | ≈ 1.0× |
| ♻️ persistent workers | RAM kept | many short epochs | ≈ 1.1× |
| 🌊 side-stream copy | a stream | copy visible in the trace | ≈ 1.0× |
| 📦 `__getitems__` | your code | per-call cost dominates | ≈ 4× more |

💡 Some rungs give **nothing**. A knob helps only if it hits the **actual bottleneck**.

<!-- _footer: "Speedups as reported by the official tutorial (its hardware is not stated; page figures should be re-checked). Your numbers will differ: the shape matters." -->

---

<!-- _class: lead -->

# 🔒 Moving bytes safely

---

## `non_blocking=True`: what it does, and what it does not

- `x.to("cuda", non_blocking=True)` can **return before the copy finishes** when `x` is pinned
- ✅ **Copy/compute overlap on the GPU needs all three:**

<div class="cols3">
<div class="card green">

📌 a **pinned** source

</div>
<div class="card blue">

🛤️ a **separate, non-default** stream

</div>
<div class="card orange">

🚚 a **free copy (DMA) engine**

</div>
</div>

- ❌ `pin_memory=True` + `non_blocking=True` on the default stream is **not** overlap
- ❌ `x.pin_memory().to(dev, non_blocking=True)` by hand can be **up to 2× slower** than a plain `.to(dev)`: `pin_memory()` blocks the host. Let the DataLoader pin.

<!-- _footer: "Official guide measures 0.375 ms for pageable.to(cuda) and 0.428 ms for pageable.pin_memory().to(cuda) on one 1M-float tensor." -->

---

## ⚠️ A race on the PCIe bus

<div class="cols">
<div>

```python
# host to device
dst = pinned.to("cuda", non_blocking=True)
pinned.zero_()        # ❌ copy may not
                      #    have happened yet

# device to host
host = result.to("cpu", non_blocking=True)
host.sum()            # ❌ may read garbage
torch.cuda.synchronize()   # ✅ wait first
```

</div>
<div class="card red">

🔁 **Week 3:** a data race.
🔁 **Week 9:** a missing barrier gives a **wrong answer, not a crash**.

Rules:
- never change a pinned source while a copy may be pending
- synchronise before you read a `non_blocking` result on the CPU

</div>
</div>

<!-- _footer: "Demo with real failures on a GPU: week10/03_pinned_transfer.py (the official guide's tests failed at iterations 2 and 0)." -->

---

<!-- _class: lead -->

# 🔍 Measure before you tune

---

## 🚦 `nvidia-smi` is a hint, not a diagnosis

- "GPU-Util 90 %" only says **some kernel was running** during that interval
- It does not say the GPU was **well used**, or **what it waited for**

Better signals:
- ⏱️ **Time the data wait** separately from the step
- 📈 **Trace** one step with `torch.profiler`
- 🖼️ **images/s** (or samples/s) for the whole loop

---

## 🕵️ Hidden syncs make a loop look input-bound

![w:860](img/w10-hidden-sync.svg)

---

## 🕵️ Which calls make the CPU wait for the GPU?

`.item()` · `.cpu()` · `print(cuda_tensor)` · `nonzero()` · `if cuda_value:` · copies back to the host

```python
total_loss += loss.item()          # ❌ waits for the GPU on every step
```

```python
running = running + loss.detach()  # ✅ stays on the GPU
if step % 50 == 0:
    print(running.item())          # ✅ look only now and then
```

The official tutorial's loop calls `loss.item()` every batch: fine for a tutorial, but **measure** it in your own loop.

<!-- _footer: "PyTorch Performance Tuning Guide: avoid operations that require synchronisation. Demo: week10/04_hidden_syncs.py (on a GPU)." -->

---

## ⏱️ Timing the wait correctly

```python
torch.cuda.synchronize(); t0 = time.perf_counter()
batch = next(it)
t1 = time.perf_counter()               # data wait: host time, no sync here
...                                    # forward, backward, optimizer step
torch.cuda.synchronize(); t2 = time.perf_counter()    # step time

data_wait += t1 - t0;  step_time += t2 - t1
```

- **Do not** synchronise before `t1`: that adds the previous step's GPU work to the data wait
- The sync before `t0` serialises CPU and GPU a little, so **the trace is the unperturbed view**

<!-- _footer: "Full loop: run_epochs() in week10/pipeline_utils.py" -->

---

## 📈 `torch.profiler`: label the wait

```python
from torch.profiler import profile, record_function, schedule, ProfilerActivity

sched = schedule(wait=1, warmup=1, active=4)       # skip 1, warm up 1, record 4
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=sched) as prof:
    for _ in range(12):
        with record_function("data_wait"):
            x, y = next(it)
        with record_function("train_step"):
            ...                                    # forward, backward, update
        prof.step()                                # one step finished

prof.export_chrome_trace("week10_trace.json")      # open at ui.perfetto.dev
```

👀 Look for long **`data_wait`** blocks and **gaps** in the GPU row.

<!-- _footer: "Official recipe: PyTorch Profiler. Full script: week10/05_profile_pipeline.py (also runs on CPU)." -->

---

## ⚙️ Worker pitfalls (read the DataLoader docs)

<div class="cols">
<div class="card blue">

### 🚀 Start method
Python **3.14+** uses `forkserver` on Linux. The dataset and `worker_init_fn` must be **picklable** (no lambdas). Guard scripts with `if __name__ == "__main__":`.

</div>
<div class="card orange">

### 🎲 Seeds and memory
Random augmentations must **differ** across workers. Big lists of Python objects can grow in every worker; **NumPy arrays** do not.

</div>
<div class="card green">

### 🏁 `in_order=False`
A slow worker no longer blocks the queue (a Week 5 straggler). The docs warn it **can harm reproducibility** and skew imbalanced data.

</div>
<div class="card red">

### 🧱 Shared memory
Batches travel through `/dev/shm`. A full one gives a **bus error** in a worker (Docker, Colab). See `set_sharing_strategy()`.

</div>
</div>

---

## 🧭 Tuning recipe

<div class="cols">
<div class="card blue">

### Always
- 📏 measure: **is the GPU waiting?**
- 🪜 climb the ladder **in order**, re-measure each time
- 🛑 stop when the loader is no longer the bottleneck

</div>
<div class="card green">

### Also (tuning guide)
- `optimizer.zero_grad(set_to_none=True)`
- `torch.backends.cudnn.benchmark = True` for fixed-size conv inputs
- fewer syncs (`.item()`)

</div>
</div>

💡 Honest outcome: *"the job is not input-bound, so none of this helps."*

---

## 🗄️ Beyond the tutorial: where the data lives

Not covered by the official tutorial, so just know the names:

- 📁 **Many small files** are slow; large **shards** read sequentially are fast (WebDataset)
- 🧮 **Parquet / Arrow** for tables, `np.memmap` for big arrays
- 💾 **Cache** decoded, resized data once instead of decoding every epoch
- 🖥️ **GPU-side** decode and augmentation (NVIDIA DALI)

See each project's own documentation when you need them.

---

## 🎓 Project 2: proposals are due this week

One page, with:

| Item | Example |
|---|---|
| **System** | Qwen2.5-0.5B-Instruct with `transformers` |
| **Workload** | 200 prompts, 256 generated tokens each |
| **Hardware** | Colab T4, PyTorch 2.x |
| **Baseline** | fp16, batch size 1, eager mode |
| **Question** | How do batch size and `torch.compile` change throughput? |
| **Interventions** (≥ 2) | batch sizes 1–16, `torch.compile`, both |
| **Metrics** | tokens/s, p95 latency, peak memory, GPU utilisation |

📌 Track A groups: say which **pipeline measurements** you will take.

<!-- _footer: "Full brief: projects/project2-hardware-aware-ai/project2_hardware-aware-ai.md" -->

---

<!-- _class: lead -->

# 🧪 Lab

---

<!-- _footer: "Scripts in week10/ · run 02_ladder.py first. Works on Colab (T4) and on a laptop; GPU-only rungs are skipped on CPU." -->

## 🧪 Lab: your GPU is idle. Find out why and fix it

| Task | File |
|---|---|
| Run the whole ladder, fill in the table | `02_ladder.py` |
| Trace the baseline and the best loader, compare the `data_wait` blocks | `05_profile_pipeline.py` |
| Pinned vs pageable copies, then the two hazards (GPU) | `03_pinned_transfer.py` |
| `.item()` every step vs every 50 (GPU) | `04_hidden_syncs.py` |

**Predict first**, then run: which rung will give the biggest gain? Which will give **nothing**?

---

## 🧾 Answer sheet

For each rung: **time**, **speedup vs the previous rung**, **cumulative speedup**, **data-wait share**.

Then complete this sentence:

> *"My job was ______-bound, because ______ ms of each ______ ms step went to waiting for data. The rung that helped most was ______. ______ gave no gain because ______."*

Also report: Python and PyTorch versions, `os.cpu_count()`, and whether the GPU was a T4.

---

## ✅ Takeaways

- 📏 **Measure the wait first.** The knobs only help an input-bound job.
- 🪜 **Climb the ladder in order and re-measure.** Some rungs give nothing.
- 📦 The biggest lever is usually **less work per sample** (batched, cached, smaller dtypes), not more workers.
- 🌊 Overlap needs a **pinned source, a side stream and a free copy engine**; `non_blocking` without a sync is a **race**.
- 🕵️ **Hidden syncs** can make an idle GPU look like a slow loader. Trust the **trace**.

---

## 🔭 Forward map and before next week

| Today | Returns in |
|---|---|
| per-GPU input throughput, `DistributedSampler` | DDP (Weeks 11–12) |
| copy/compute overlap, hidden syncs | DDP communication overlap, mixed precision, `torch.compile` |
| the measurement recipe | Project 2 reports (every track) |

**Before next week**
- ✍️ Finish the lab answer sheet; submit the **Project 2 proposal**
- 📖 Read the official *Data Loading Optimization in PyTorch* tutorial
- 🤔 With **2 GPUs**, how does each one get *different* data?

---

## 📚 References

- PyTorch tutorial: [Data Loading Optimization in PyTorch](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)
- PyTorch docs: [`torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html) · [Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- PyTorch guide: [Good usage of `non_blocking` and `pin_memory()`](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
- PyTorch recipe: [Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) · viewer: [ui.perfetto.dev](https://ui.perfetto.dev)
- [Course repository](https://github.com/adaskin/hpc-for-ai-course) · scripts in `lectures/week10/`

Figures `img/w10-*.svg` are drawn by `week10/make_figures.py`; see `img/CREDITS.md`.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank You

### Next week: many GPUs, data parallelism with DDP (Week 11)

*YZM 513 – İstanbul Medeniyet Üniversitesi – Fall 2026*
