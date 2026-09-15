# YZM 513 – High-Performance Programming Techniques for Artificial Intelligence

**Program:** Yapay Zekâ Mühendisliği Tezli Yüksek Lisans (AI Engineering, M.Sc.)  
**Department:** Bilgisayar Mühendisliği Anabilim Dalı Başkanlığı  
**University:** İstanbul Medeniyet Üniversitesi  
**Instructor:** Ammar Daşkın  
**Semester:** Fall 2026 – 2027  
**Format:** Face-to-face, lecture + in-class labs  
**Expected enrolment:** ~10 students    
*Prepared with Qwen AI*  

---

## 1  Course Description

Modern AI and deep-learning workloads routinely exceed what a single
CPU core can deliver. This course introduces the full stack of
high-performance computing (HPC) techniques that AI engineers use in
practice: from profiling and vectorising Python code, through CPU-level
parallelism, GPU programming, data-pipeline optimisation, to distributed
training of large models.

The course is designed so that **no prior parallel-programming or
GPU-programming background is assumed**. Foundational concepts
(processes, threads, memory hierarchies, SIMT execution) are introduced
from scratch before being applied to AI workloads. All hands-on work
stays in the **Python ecosystem** (NumPy, Numba, CuPy, PyTorch); no
prior C/C++ or raw CUDA-C experience is required.

> **A note on approach.** The GPU-programming portion (Weeks 5–8) is a
> rapidly evolving area. The instructor and students will work through
> the material together, using live coding, Colab notebooks, and
> profiling tools. The goal is a solid working understanding, not
> exhaustive CUDA mastery.

---

## 2  Course Objectives

By the end of the semester students will be able to:

1. Profile Python code, identify bottlenecks (including GIL effects),
   and apply vectorisation and JIT-compilation fixes.
2. Use threading, multiprocessing, and modern GIL-free runtimes
   (Python 3.13 +) for CPU-level parallelism.
3. Explain GPU architecture and write basic parallel kernels using
   CuPy and Numba CUDA.
4. Optimise data-loading pipelines inside deep-learning frameworks
   (primarily PyTorch).
5. Apply distributed-training paradigms (DDP / FSDP, model parallelism,
   DeepSpeed ZeRO) and hardware-aware techniques (mixed precision,
   quantisation) to scale AI model training.

---

## 3  Prerequisites / Assumed Background

| Expected | Not required |
|----------|--------------|
| Comfortable Python programming | Parallel / concurrent programming |
| Basic data structures & algorithms | OS internals (synchronisation, scheduling) |
| Introductory machine-learning course | CUDA, C/C++, GPU programming |
| Basic linear algebra | Distributed systems |

If you have taken **Operating Systems** or equivalent, great—but
it is not mandatory. Week 1 includes a short, ungraded diagnostic so the
instructor can calibrate depth.

---

## 4  Learning Outcomes

| ID | Learning Outcome |
|----|------------------|
| LO-1 | Analyse performance bottlenecks in Python (including GIL effects and its current status) and utilise profiling tools. |
| LO-2 | Optimise code using vectorisation, JIT compilation (Numba), and multi-threading / multi-processing at the CPU level. |
| LO-3 | Understand GPU (accelerator) architecture and develop basic parallel kernels using CuPy and Numba CUDA. |
| LO-4 | Maximise data-pipeline performance and hardware utilisation within deep-learning frameworks (PyTorch). |
| LO-5 | Apply distributed-training paradigms (DDP / FSDP, Model / Tensor Parallelism) and hardware-aware optimisations (mixed precision, quantisation) to scale AI model training. |

---

## 5  Weekly Schedule

> **Legend.** 📖 = suggested pre-reading · 🛠 = in-class lab / hands-on
> component. All CUDA labs are designed to run on **Google Colab**
> (free T4 GPU) so no personal NVIDIA GPU is required.

### Week 1 – Why Performance Matters; Profiling Python
- AI training costs: FLOP counts, memory-bandwidth walls, GPU-hours.
- Python's performance ceiling; CPython execution model at a 10 000-ft view.
- Profiling tools: `cProfile`, `line_profiler`, `timeit`, `py-spy`.
- **Concept intro:** process vs. thread; why parallelism helps.
- 🛠 Profile a deliberately slow script; identify top-3 bottlenecks.
- 📖 Install course environment (Conda / venv, Python ≥ 3.12, NumPy,
  Numba, PyTorch). Setup guide posted on the course site.

### Week 2 – Vectorisation & JIT: NumPy + Numba
- NumPy memory model: contiguous arrays, broadcasting, avoiding
  Python-level loops.
- Numba `@jit` / `@njit`: compiling Python to machine code.
- When vectorisation is enough vs. when you need JIT.
- 🛠 "Make this 100× faster" exercise: start with a nested-loop
  Python function → NumPy → Numba; benchmark each step.
- 📖 NumPy "Internals" docs; Numba 5-min guide.

### Week 3 – CPU Parallelism: Threading, Multiprocessing & the GIL
- `concurrent.futures`: `ThreadPoolExecutor` vs. `ProcessPoolExecutor`.
- The GIL: what it blocks, what it doesn't (I/O-bound vs. CPU-bound).
- Python 3.13+ free-threaded build (PEP 703): what changes, what
  doesn't. Subinterpreters in brief.
- 🛠 Lab: I/O-bound task (parallel web fetches) with threads vs.
  CPU-bound task with processes; optional GIL-disabled 3.13 comparison.
- 📖 PEP 703 summary; `concurrent.futures` docs.

### Week 4 – Task Parallelism at Scale: Dask & Ray (Light Touch)
- Dask arrays / dataframes: "NumPy that doesn't fit in RAM."
- Ray: task-parallel runtime; connection to Ray Train / Ray Tune.
- When to reach for these vs. plain multiprocessing.
- 🛠 Demo: out-of-core computation with Dask on a dataset larger than
  RAM. (No Spark.)
- 📖 Dask 10-min quickstart; Ray "Walkthrough" page.

### Week 5 – GPU Architecture & First Steps with CuPy
- CPU vs. GPU design goals: latency-oriented vs. throughput-oriented.
- SIMT execution model; grid → block → thread hierarchy (conceptual).
- GPU memory overview (global, shared, registers) – qualitative.
- CuPy as "NumPy on the GPU": array ops, `cupy.asnumpy()`.
- 🛠 Run CuPy array operations on Colab (T4); time them against NumPy.
- 📖 NVIDIA GPU architecture whitepaper (skim); CuPy quickstart.

### Week 6 – Writing Your First CUDA Kernel (Numba CUDA)
- Mapping a loop to a kernel: thread indexing, `cuda.grid(1)`.
- Writing, launching, and synchronising a Numba CUDA kernel.
- Memory transfers: host ↔ device; `cuda.to_device()`.
- 🛠 Vector-addition kernel → extend to element-wise ReLU; compare
  against CuPy built-in.
- 📖 Numba CUDA tutorial (sections 1–4).

### Week 7 – Midterm Review
- Guided review of Weeks 1–6.
- Open Q&A; practice profiling & short coding problems.
- No new material.

### Week 8 – CUDA Memory Hierarchy & Profiling
- Global vs. shared vs. local memory; coalesced access.
- Case study: naïve matrix-multiply kernel → tiled kernel with shared
  memory → compare with CuPy / cuBLAS built-in.
- NVIDIA profiling tools: `nsight-compute` (or `nvprof`); PyTorch
  profiler for GPU ops.
- 🛠 Profile the tiled-matmul kernel; identify the bottleneck; write a
  short benchmark report.
- 📖 Kirk & Hwu, Ch. 4–5 (memory hierarchy).

### Week 9 – Data Pipelines for Deep Learning
- `torch.utils.data.DataLoader`: `num_workers`, `pin_memory`,
  `prefetch_factor`, `persistent_workers`.
- Streaming formats: WebDataset / TFRecord (brief).
- Diagnosing GPU idle time: `torch.profiler` + TensorBoard trace view.
- 🛠 Exercise: "Your GPU is idle 60 % of the epoch—find out why and
  fix it" on a provided training script.
- 📖 PyTorch DataLoader docs; PyTorch profiler tutorial.

### Week 10 – Distributed Training I: Data Parallelism
- Data-parallel concept: gradient all-reduce.
- PyTorch `DistributedDataParallel` (DDP): launch, rank, world size.
- PyTorch FSDP (Fully Sharded Data Parallel) – when and why.
- 🛠 Run a 2-process DDP job on a single machine (gloo backend, CPU)
  or on Colab with a small model.
- 📖 PyTorch DDP tutorial; FSDP "Getting Started" doc.

### Week 11 – Distributed Training II: Model Parallelism & DeepSpeed
- Pipeline parallelism: micro-batches, stage scheduling.
- Tensor parallelism (Megatron-style): splitting weight matrices.
- DeepSpeed ZeRO Stages 1 / 2 / 3: what gets sharded.
- 🛠 Walk through a DeepSpeed config file; inspect a profiler trace
  of a ZeRO-3 run (pre-recorded if multi-GPU unavailable).
- 📖 ZeRO paper (Rajbhandari et al., 2020); DeepSpeed "Getting
  Started."

### Week 12 – Hardware-Aware Training: Mixed Precision & Quantisation
- FP32 → FP16 / BF16: why it works, loss-scaling.
- `torch.amp.autocast` + `GradScaler` in practice.
- FP8 training (Transformer Engine) – current frontier.
- Post-training INT8 quantisation: concepts + quick demo.
- 🛠 Train a small CNN in FP32 vs. BF16; compare wall-clock, memory,
  and loss curves.
- 📖 PyTorch AMP tutorial; "A Survey of Quantization" (short).

### Week 13 – Inference Optimisation & Serving
- `torch.compile` (TorchDynamo / Inductor): what it does, speed-ups.
- TensorRT / ONNX Runtime for deployment (conceptual + demo).
- KV-cache, continuous batching for LLM serving (vLLM overview).
- 🛠 Benchmark a small model: eager vs. `torch.compile` vs. exported
  ONNX.
- 📖 `torch.compile` docs; vLLM blog post.

### Week 14 – Project Presentations (Part 1)
- Group presentations (~15 min each): problem, approach, benchmark
  methodology, results, discussion.
- Live demo encouraged.
- Peer Q&A.

### Week 15 – Project Presentations (Part 2) & Wrap-Up
- Remaining group presentations.
- Course retrospective: what worked, what to explore next.
- Pointers to further study (custom CUDA kernels, Triton, multi-node
  training, emerging accelerators).

---

## 6  Assessment

| Component | Weight | Details |
|-----------|--------|---------|
| **Term Project** | 30 % | Groups of ~3. Choose an AI workload, profile it, apply ≥ 2 HPP techniques from the course, and produce a benchmark report + live demo. Rubric posted by Week 6. |
| **Midterm (Vize)** | 30 % | Covers Weeks 1–6. Mix of short-answer concepts and a hands-on profiling / optimisation task. |
| **Final Exam** | 40 % | Covers Weeks 8–13 with emphasis on GPU, pipelines, and distributed training. Open-book component for distributed-training scenarios. |

### Project Guidelines (summary)
- **Deliverables:** 6–10-page report (problem statement, baseline
  profiling, optimisations applied, before/after benchmarks, discussion)
  + 10-min presentation + live or recorded demo.
- **Benchmarking requirement:** report must include wall-clock time,
  peak memory, and GPU utilisation for at least the baseline and the
  optimised version.
- **Example project themes:**
  - Optimise a PyTorch training pipeline end-to-end (DataLoader + AMP
    + `torch.compile`).
  - Implement and benchmark a CUDA kernel (e.g., custom attention)
    against the PyTorch built-in.
  - Compare DDP vs. FSDP vs. DeepSpeed ZeRO-2 for a given model.
  - Profile and optimise an LLM inference server (batching, KV-cache,
    quantisation).

---

## 7  Readings & Resources

### Primary
| Resource | Used for |
|----------|----------|
| **Numba CUDA documentation** | Weeks 5–8 |
| **CuPy User Guide** | Weeks 5–6 |
| **PyTorch official docs & tutorials** (DataLoader, DDP, FSDP, AMP, `torch.compile`, Profiler) | Weeks 9–13 |
| **DeepSpeed documentation + ZeRO paper** (Rajbhandari et al., SC 2020) | Week 11 |

### Supplementary
- Kirk & Hwu, *Programming Massively Parallel Processors*, 4th ed.
  (Ch. 1–5 for GPU architecture & memory background).
- NVIDIA *CUDA C++ Programming Guide* – reference only; not required
  reading.
- Selected short papers / blog posts posted on the course site
  (Megatron-LM tensor parallelism, vLLM PagedAttention, etc.).

### Tools used in the course
Python ≥ 3.12 · NumPy · Numba · CuPy · PyTorch ≥ 2.x ·
Google Colab (free tier) · `line_profiler` / `py-spy` ·
`torch.profiler` / TensorBoard · DeepSpeed · Dask / Ray (Week 4 only)

---

## 8  Environment Setup (Before Week 1)

Students should arrive with:

1. **Local machine:** Conda (Miniconda) or `venv` with Python ≥ 3.12,
   NumPy, Numba, PyTorch (CPU build is fine for Weeks 1–4).
2. **Google Colab account** (free) for GPU labs (Weeks 5 onward).
3. A GitHub account – assignments and project code will be managed via
   a course GitHub organisation / classroom.



---

## 9  Course Policies

- **Attendance.** Face-to-face. With a few students, missing a session is
  noticeable; please notify the instructor in advance if you must be
  absent. Lab components are difficult to replicate asynchronously.
- **Collaboration.** Discussion is encouraged.