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
  section.lead {
    text-align: center;
  }
  section.lead h1 {
    font-size: 52px;
  }
  table {
    font-size: 24px;
  }
  code {
    font-size: 22px;
  }
  pre {
    font-size: 20px;
  }
  img {
    display: block;
    margin: 0 auto;
  }
  footer {
    font-size: 16px;
    opacity: 0.5;
  }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# YZM 513 – High-Performance Programming for AI

### Week 6: Inside the GPU
### Hardware, the CUDA Stack, and First Steps with CuPy

**İstanbul Medeniyet Üniversitesi**
AI Engineering M.Sc. – Fall 2026

Ammar Daşkın

---

## Where we are in the course

```text
Week 1–2:  One core, made fast        (vectorise, JIT, BLAS)
Week 3–4:  One machine, many cores    (threads, processes, async, pools)
Week 5:    Many machines              (Dask, Ray, cost of distribution)
Week 6:    One GPU, thousands of cores  ← TODAY: hardware + CuPy
Week 7:    Midterm (covers Weeks 1–6)
Week 8:    Writing GPU kernels        (Numba CUDA)
Week 9:    Feeding the GPU            (DataLoader pipelines)
Week 10–11: Many GPUs                 (DDP, model parallelism)
```

> Every parallel system answers the same four questions — **split, assign, move, wait**.
> The GPU changes the *scale* (thousands of cores, not 8) and the *memory model* (you move data explicitly).

---

<!-- _class: lead -->

# Part I
# Why GPUs?

---

## The three walls, revisited

| Wall | Weeks 1–2 answer | Week 5 answer | **This week** |
|------|-----------------|---------------|---------------|
| **Language wall** | vectorise, Numba | — | still true on GPU |
| **Core wall** | threads, processes | Dask, Ray | **GPU: thousands of cores in one chip** |
| **Memory wall** | cache-friendly code | partitions, object stores | **separate VRAM — data must cross PCIe** |

> AI hit the core wall *hard*: training GPT-class models on CPUs would take millennia.
> GPUs are the industry's answer: a chip built to do one thing — massive parallel arithmetic — extremely well.

---

## Concrete numbers

| | CPU (e.g. Xeon/EPYC) | GPU (Colab T4) | GPU (RTX 4090) |
|---|---|---|---|
| "cores" | 8–64 big | 2,560 CUDA + 320 Tensor | 16,384 CUDA + 512 Tensor |
| Clock | 3–5 GHz | 1.6 GHz | 2.5 GHz |
| Memory | 100 GB/s (DDR5) | 320 GB/s (GDDR6) | 1,008 GB/s (GDDR6X) |
| Peak FP32 | 1–3 TFLOPS | ~8 TFLOPS | ~83 TFLOPS |
| Price class | — | free on Colab | ~$1,600 |

> Individually, each GPU core is *weaker* than a CPU core. The GPU wins by fielding
> **thousands at once**. RTX 4090 vs top CPU: ~30–50× peak FP32.

---

## Why AI specifically loves GPUs

Deep learning is, mathematically, mostly **matmul**:

- One linear layer: `Y = XW + b` — a single matmul
- Attention: batched matmul of Q·Kᵀ, then ·V
- Convolutions: matmuls with im2col (cuDNN does this for you)
- Backprop: more matmuls

A matmul of (4096×4096)·(4096×4096) is **137 GFLOP of independent multiply-adds** —
the purest data-parallel workload imaginable. That is exactly what a GPU is built to execute.

<!-- _footer: "FLOP counting reminder from Week 1: 2n³ for n×n matmul." -->

---

<!-- _class: lead -->

# Part II
# Hardware Background

---

## Two design philosophies

**CPU — minimise latency for one thread:**
big cores, deep out-of-order pipelines, branch prediction, 30–100 MB cache.
Spend transistor budget making *one* stream of instructions finish fast.

**GPU — maximise throughput across thousands of threads:**
small cores, shallow pipelines, tiny per-SM cache, no branch prediction.
Spend transistor budget on *more arithmetic units*.

> Same silicon budget, opposite bet: **latency machine vs. throughput machine.**

---

![w:980](img/w06-cpu-vs-gpu-die.svg)

---

## Inside the GPU: the SM (Streaming Multiprocessor)

- The GPU is a collection of **SMs** (T4: 40, A100: 108, H100: 132, RTX 5090: 170)
- Each SM contains:
  - **CUDA cores** — simple FP32/INT ALUs (128 per SM on recent designs)
  - **Tensor Cores** — fused matmul units (FP16/BF16/FP8/FP4) — the "AI" units
  - **Scheduler(s)** that pick warps to run each cycle
  - small **shared memory** + register file (Week 8 territory)
- A hardware **GigaThread engine** assigns blocks to SMs as they become free

<!-- _footer: "NVIDIA CUDA C Programming Guide, ch. 2: https://docs.nvidia.com/cuda/cuda-c-programming-guide/" -->

---

## Tensor Cores: why the AI numbers are insane

CUDA cores do one FP32 FMA per cycle. **Tensor Cores do a whole small matmul per cycle.**

| Generation | Flagship | Tensor peak | New capability |
|---|---|---|---|
| Volta (2017) | V100 | 125 TFLOPS FP16 | 1st Tensor Core |
| **Turing (2018)** | **T4 ← our Colab GPU** | ~65 TFLOPS FP16 | INT8/INT4 inference |
| Ampere (2020) | A100 | 312 TFLOPS TF32 | BF16, sparsity |
| Hopper (2022) | H100 | 1,979 TFLOPS FP16 | **Transformer Engine (FP8)** |
| Blackwell (2024–25) | B200 | 2,250 TFLOPS FP16, **9,000 TFLOPS FP4** | FP4, 208 B transistors |

> Peak tensor throughput ≠ usable throughput — Week 12 (mixed precision) revisits this.

---

## GPU generations timeline

![w:1050](img/w06-gpu-generations.svg)

---

## GPU memory technology: GDDR vs HBM

| | GDDR (graphics card) | HBM (datacenter) |
|---|---|---|
| Where | RTX 4090/5090, T4 | A100, H100, B200 |
| Trick | very fast signalling, wide-ish bus | **stacked dies** next to GPU (2.5D packaging), 4096+ bit bus |
| Bandwidth | 0.3–1.8 TB/s | 2.0–8 TB/s |
| Cost / capacity | cheap, 16–32 GB | expensive, 80–192 GB |

> For LLM *inference*, memory bandwidth is usually the bottleneck: every generated token
> must stream all weights from VRAM. That is why token/s scales with GB/s, not TFLOPS.

---

## Getting data between chips: PCIe vs NVLink

| Link | Bandwidth | Used for |
|---|---|---|
| PCIe 4.0 x16 | ~25–32 GB/s | CPU ↔ GPU (your data uploads) |
| PCIe 5.0 x16 | ~50–64 GB/s | newer CPU ↔ GPU |
| NVLink 4 (Hopper) | 900 GB/s | GPU ↔ GPU inside a node |
| NVLink 5 (Blackwell) | 1.8 TB/s | GPU ↔ GPU inside a node |

> **Rule of the week:** PCIe is ~10–100× slower than the GPU's own memory.
> Upload once, compute a lot, download once.

---

<!-- _class: lead -->

# Part III
# Core Technologies: the CUDA Stack

---

![w:820](img/w06-software-stack.svg)

---

## The stack, bottom-up

**CUDA (Compute Unified Device Architecture)** — NVIDIA's platform:
- programming model (kernels, threads, memory) + compiler (`nvcc`) + runtime + drivers
- you almost never touch it directly from Python — libraries call it for you

**CUDA libraries** — the "BLAS of the GPU" (cf. Week 2!):
- **cuBLAS** — matmul, GEMM (what `a @ b` calls on GPU)
- **cuDNN** — convolutions, attention, normalization primitives for deep learning
- **NCCL** — collective communication for multi-GPU (Weeks 10–11)
- cuFFT, cuSPARSE, cuSOLVER, Thrust…

**Frameworks** — CuPy, PyTorch, JAX, TensorRT (inference compiler), Triton (GPU compiler)

<!-- _footer: "https://developer.nvidia.com/gpu-accelerated-libraries" -->

---

## Why this matters for you

When you write `a @ b` in CuPy or `torch.matmul`, the call chain is roughly:

```text
your Python code
  → CuPy / PyTorch dispatcher
    → cuBLAS GEMM (vendor-tuned kernel, often handwritten in CUDA/PTX)
      → CUDA runtime → driver → SMs + Tensor Cores
```

> You have been benefiting from this stack all course: NumPy → OpenBLAS/MKL on CPU is the
> *same pattern*. GPU = same idea, another backend. This is why CuPy can be a drop-in NumPy.

---

<!-- _class: lead -->

# Part IV
# The SIMT Execution Model

---

## SIMT = Single Instruction, Multiple Threads

One kernel launch creates a **grid** of **blocks** of **threads**:

![w:1000](img/w06-simt-hierarchy.svg)

- Each **thread** runs the same function (the *kernel*) on its own data
- Threads in a block can cooperate (shared memory) and are co-scheduled on one SM
- Blocks are scheduled independently → the GPU scales from 40 to 170 SMs without code changes

<!-- _footer: "Diagram modelled on NVIDIA CUDA C Programming Guide, 'Thread Hierarchy'." -->

---

## Thread indexing: how each thread finds its element

```python
# One thread per output element — you write what ONE thread does:
i = blockIdx.x * blockDim.x + threadIdx.x
c[i] = a[i] + b[i]
```

Grid 4 × blockDim 8 → 32 threads → elements 0..31 handled simultaneously.

> **Connection to Week 3:** data parallelism ("same op, different data") *is* SIMT.
> What took you `Pool(8)` on CPU, the GPU does with `Pool(2560)` — in hardware.

---

## Warps: the unit of actual execution

- Threads run in **warps: groups of 32** that execute in lockstep on an SM
- The scheduler runs several warps per SM and **switches for free** when one stalls on memory
  → latency hiding: warps waiting on HBM don't block warps that are ready
- **Divergence warning:** if threads in a warp take different branches (`if/else`),
  *both* paths execute serially → wasted cycles

> For today's library-level GPU use (CuPy) you don't manage warps — but in Week 8 kernels
> this will be your first optimisation lever.

---

## Map AI ops onto threads

| AI operation | Parallelisation |
|---|---|
| ReLU on 1M activations | 1M threads: `max(0, x)` each |
| Matmul 4096³ | one thread per output element = 16.7M threads (≈ 65k blocks) |
| Attention scores | one thread per (query, key) pair |
| Batch of 256 images | extra grid dimension |

> Mental shift: **you don't write a loop over elements. You write what ONE thread does,
> and the hardware runs millions of copies.**

---

<!-- _class: lead -->

# Part V
# GPU Memory: Two Worlds, One Cable

---

![w:1000](img/w06-memory-transfer.svg)

---

## The transfer rules

1. CPU RAM and GPU VRAM are **separate address spaces** — no sharing (by default)
2. Every upload/download crosses PCIe: **~25–55 GB/s** vs GPU memory at **300–8,000 GB/s**
3. Kernels are **asynchronous**: launch returns immediately; results appear "later"
4. All CuPy intermediates stay on GPU — transfer **in once, out once**

**Amortisation rule (Week 5's cost ladder, revisited):**
GPU wins when `compute_time ≫ transfer_time`. For an n×n matmul:

- transfer time grows like **n²** (three n×n matrices cross PCIe)
- compute time grows like **n³** (2n³ FLOPs on-device)

Small n → transfer dominates → **CPU wins**. Large n → GPU wins by 10–100×.

---

<!-- _class: lead -->

# Part VI
# CuPy: NumPy on the GPU

---

## The API mapping

| NumPy | CuPy |
|---|---|
| `np.array(...)`, `np.zeros(...)` | `cp.array(...)`, `cp.zeros(...)` |
| `a @ b`, `a.sum()`, slicing | **identical syntax** |
| `np.linalg.svd` | `cp.linalg.svd` |
| result on CPU | result on GPU until `.get()` |
| `float32`/`float64` dtypes | prefer `float32` (native GPU speed) |

```python
import numpy as np, cupy as cp

a_gpu = cp.asarray(np.random.rand(4096, 4096).astype(np.float32))  # upload once
b_gpu = a_gpu @ a_gpu                # runs on GPU via cuBLAS
c_gpu = cp.maximum(b_gpu, 0)         # element-wise kernel, stays on GPU
result = c_gpu.get()                 # download once
```

---

## Live demo (Colab, T4)

```bash
!nvidia-smi          # which GPU, memory, utilisation
!pip install -q cupy-cuda12x
```

Benchmark matmul at 256, 1024, 4096, 8192 — NumPy vs CuPy, with and without transfer:

| n | NumPy time | CuPy compute | + transfer |
|---|---|---|---|
| 256 | ~0.2 ms | ? | ? |
| 1024 | ~8 ms | ? | ? |
| 4096 | ~0.5 s | ~2 ms | ? |
| 8192 | ~4 s | ~15 ms | ? |

> Your job in the lab: find the **crossover point** where GPU pays off — and explain it with the cost ladder.

---

## CuPy vs PyTorch — when to use which

| | CuPy | PyTorch |
|---|---|---|
| Purpose | GPU NumPy | deep learning framework |
| Autograd | ❌ | ✅ |
| GPU backend | cuBLAS, cuDNN | cuBLAS, cuDNN |
| This course | Weeks 6–7 (learn the GPU) | Weeks 9+ (training, pipelines, DDP) |

- Use CuPy: preprocessing, custom array ops, quick GPU prototyping, porting NumPy code
- Use PyTorch: anything that must train a neural network

---

<!-- _class: lead -->

# Part VII
# Lab (Colab)

---

## Exercises (~25 min, fill the results table)

1. `!nvidia-smi` — identify GPU, VRAM, driver
2. **Matmul sweep**: time `np.dot` vs `cp.dot` at n = 256…8192 → find crossover
3. **Element-wise**: ReLU / sigmoid on 10M elements, CPU vs GPU
4. **The transfer trap**: time `cp.asarray` alone vs the compute — when does transfer dominate?
5. **Bonus**: `cp.linalg.svd` vs `np.linalg.svd` on a big matrix (Project 1 callback!)

**Deliverable:** completed table + one sentence: *"GPU is worthwhile on T4 when … because …"*

---

<!-- _class: lead -->

# Part VIII
# Midterm Review (Weeks 1–6)

---

## Coverage and key formulas

| Week | Must know | Week | Must know |
|---|---|---|---|
| 1 | profiling, FLOPS vs GB/s | 4 | GIL, pools, async, I/O vs CPU-bound |
| 2 | vectorise, broadcast, `@njit` | 5 | partitioning, granularity, Dask vs Ray, comm cost |
| 3 | threads/processes, races, locks | 6 | CPU vs GPU, SIMT, transfer, CuPy basics |

$$S(N)=\frac{1}{f_s+\frac{1-f_s}{N}} \qquad S_{comm}(N)=\frac{1}{f_s+f_c+\frac{1-f_s-f_c}{N}} \qquad \text{GFLOP/s}=\frac{2n^3}{t\cdot10^9}$$

**Format:** short answers + one code-analysis + one design question · ~90 min · formula sheet provided

---

## Five practice questions

1. 10M-element array, Python loop is the bottleneck → what do you try, in order?
2. GPU utilisation 30% during training, DataLoader slow → problem and fix?
3. Why does `ThreadPoolExecutor` help NumPy ops but not pure-Python loops? (GIL)
4. 4096² float32 matrix: VRAM used? Transfer time over PCIe 4.0? (64 MB, ~2.5 ms)
5. Dask with 2000 tiny tasks slower than serial — why? (granularity vs overhead)

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Sources & Further Reading

**Hardware**
- CUDA C Programming Guide (thread hierarchy, memory): docs.nvidia.com/cuda/cuda-c-programming-guide
- Blackwell architecture: nvidia.com/en-us/data-center/technologies/blackwell-architecture
- DGX B200 specs: nvidia.com/en-us/data-center/dgx-b200
- RTX 5090 specs: nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090
- GPU list by CUDA capability: developer.nvidia.com/cuda-gpus

**Software**
- CuPy docs: docs.cupy.dev · NVIDIA GPU-accelerated libraries: developer.nvidia.com/gpu-accelerated-libraries
- Colab GPU FAQ: research.google.com/colaboratory/faq

Figures `w06-*.svg` are course-generated (see img/CREDITS.md).
