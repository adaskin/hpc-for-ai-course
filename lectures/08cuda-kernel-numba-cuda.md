---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 24px;
  }
  section.lead {
    text-align: center;
  }
  section.lead h1 {
    font-size: 52px;
  }
  table {
    font-size: 20px;
  }
  code {
    font-size: 18px;
  }
  pre {
    font-size: 17px;
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

### Week 8: Writing Your First CUDA Kernel
### From CuPy to Numba CUDA

**İstanbul Medeniyet Üniversitesi**
AI Engineering M.Sc. – Fall 2026

Ammar Daşkın

> **Schedule note:** this is the original syllabus Week 6 topic, moved to **Week 8** after the midterm.

---

## Learning Objectives

By the end of this lecture and lab, you can:

1. Explain **host, device, kernel, launch, grid, block, thread, warp, and SM**.
2. Describe how a kernel launch becomes blocks scheduled onto SMs.
3. Calculate 1D and 2D launch geometry and global thread indices safely.
4. Move data between host and device and synchronize correctly.
5. Write, verify, and benchmark basic Numba CUDA kernels.
6. Decide when a custom kernel is justified—and when a library is better.

> **Prerequisite:** Week 6's conceptual GPU architecture and CuPy array operations.

---

## Welcome Back: Where We Are

**Before midterm (Weeks 1–6):**
- Profile → vectorise → parallelise → distribute
- GPU architecture: SMs, CUDA cores, warps, SIMT
- CuPy: GPU computing with NumPy-like syntax

**After midterm (Weeks 8–13):**
- Write GPU kernels yourself (this week + next)
- Feed the GPU efficiently (Week 10)
- Train across GPUs (Weeks 11–12)
- Deploy efficiently (Week 13)

> Week 6: you *used* the GPU through CuPy.
> This week: you *program* the GPU directly.

---

## The Week in One Question

> CuPy already runs kernels for you. Why write your own?

Three reasons:

| Reason | Example |
|--------|---------|
| **Custom operations** | A new activation or fused residual operation not available as one library call |
| **Kernel fusion** | Combine several element-wise ops to avoid full-size intermediates and extra memory traffic |
| **Understanding** | You can't optimise what you don't understand. Week 9 profiles kernels. |

> `torch.compile` (Week 13) fuses kernels automatically.
> But you need to understand what fusion *does* to know when it helps.

---

## Today's Arc

| Part | Content | Time |
|------|---------|:----:|
| I | Why write kernels (motivation + recap) | 10 min |
| II | General CUDA programming model + indexing | 30 min |
| III | Numba CUDA: your first kernel | 25 min |
| IV | Memory transfers and device arrays | 15 min |
| *Break* | | 10 min |
| V | AI examples: ReLU, softmax, attention scores | 20 min |
| VI | Performance: when your kernel wins (and loses) | 10 min |
| VII | Connection to the course + lab | 10 min |
| **Total** | | ~130 min |

---

<!-- _class: lead -->

# Part I
# Why Write Kernels?

---

## What CuPy Does Under the Hood

When you wrote this in Week 6:

```python
c_gpu = a_gpu @ b_gpu       # one line
```

The call chain was:

```text
Your Python
  → CuPy dispatcher
    → cuBLAS GEMM kernel (hand-tuned, thousands of lines of CUDA)
      → CUDA runtime → driver → SMs
```

You got a fast matmul, but you cannot rewrite its internal thread mapping,
fuse its next operation into the same kernel, or change its memory-access pattern.

> This week: you write the kernel. You control the mapping and per-thread code.

---

## When You Need Your Own Kernel

| Scenario | Why CuPy isn't enough |
|----------|----------------------|
| Custom activation: `x * sigmoid(x)` (SiLU/Swish) | Exists in PyTorch, but what about a *new* activation you invented? |
| Fused bias + activation + residual add | Separate element-wise kernels create intermediates and extra memory traffic. |
| Custom attention mask | Standard softmax exists; your sparse mask pattern doesn't. |
| Preprocessing on GPU | Decode JPEG → normalise → augment, all without leaving VRAM. |

> **Rule:** Use library kernels when they exist. Write your own when:
> (a) the operation doesn't exist, or
> (b) fusing multiple operations saves memory bandwidth.

---

## Kernel Fusion: The Key Motivation

From Week 6's roofline: element-wise ops are often **bandwidth-bound**.

```text
Unfused: three kernels and materialised intermediates
  t1 = x + bias       → read x, bias; write t1
  t2 = relu(t1)       → read t1;     write t2
  out = t2 + residual → read t2, residual; write out

Fused: one kernel, no full-size t1 or t2
  out = relu(x + bias) + residual
```

Benefits:
- Less global-memory traffic and fewer kernel-launch overheads.
- No large intermediate arrays to allocate and revisit.

> The speedup is **not automatically “3×”**: measure it.
> Normalisation and softmax also require reductions, so they cannot be fused as
> a trivial one-pass element-wise chain without a cooperation strategy.

---

## Recap: Architecture vs. Programming Model

| Hardware view (Week 6) | Software view (this week) |
|---|---|
| Streaming multiprocessor (**SM**) | Thread **block** is assigned to one SM |
| CUDA cores / functional units | Threads perform arithmetic and memory operations |
| Warp scheduler | Hardware schedules warps from active blocks |
| Device memory and on-chip memory | Global, shared, and per-thread storage |

- You write the work for **one thread**.
- A launch creates a **grid of blocks**, not one fixed block per SM.
- Blocks may run in any order and in waves; launched threads are **not all simultaneously resident**.

> CUDA separates what you express (a grid of work) from how hardware schedules it.

---

<!-- _class: lead -->

# Part II
# The General CUDA Programming Model

---

## CUDA in One Picture

**CUDA is a heterogeneous programming model:** the CPU and GPU cooperate.

```text
HOST (CPU)                                      DEVICE (GPU)
Python / C++ starts the program
  │ allocate + initialise host data
  │ copy inputs to device memory
  │ launch kernel ───────────────────────────▶  a grid of threads runs
  │ optionally do other host work               the same kernel function
  │ synchronize / copy results ◀──────────────  writes output to memory
  ▼ verify, continue, or launch more work
```

- **Host:** CPU and its directly attached memory.
- **Device:** GPU and its directly attached memory.
- **Kernel:** a function launched for device execution.
- **Launch:** creates many threads executing that kernel.

---

## Conceptual Hardware: Host, Device, and SMs

![h:395](img/w08-gpu-cpu-system.png)

*Source: NVIDIA CUDA Programming Guide, Fig. 2. Interconnect and sizes vary by system.*

---

## A CUDA Program in Six Steps

| Step | Where | Typical Numba CUDA operation |
|:---:|---|---|
| 1 | Host | Create NumPy inputs and choose `N` |
| 2 | Host → device | `cuda.to_device(a_np)` |
| 3 | Host | Allocate output with `cuda.device_array(...)` |
| 4 | Host launches | `kernel[blocks, threads](...)` |
| 5 | Device | Each thread computes an index and writes output |
| 6 | Device → host | `out_d.copy_to_host()`; verify with NumPy |

> A kernel does **not return a Python value**. It communicates by writing to memory.

---

## Software Hierarchy: Grid → Block → Thread

![h:315](img/w08-grid-of-thread-blocks.png)

*Source: NVIDIA CUDA Programming Guide, Fig. 3; arrows represent threads.*

---

## The Launch Geometry Vocabulary

| Object | Question it answers | Numba CUDA name |
|---|---|---|
| `gridDim` | How many blocks in each grid dimension? | `cuda.gridDim.x/y/z` |
| `blockIdx` | Which block is this? | `cuda.blockIdx.x/y/z` |
| `blockDim` | How many threads in each block dimension? | `cuda.blockDim.x/y/z` |
| `threadIdx` | Which thread am I inside this block? | `cuda.threadIdx.x/y/z` |

Every block has the **same dimensions**; grids and blocks can be **1D, 2D, or 3D**.

**Total launched threads (1D):**

```text
gridDim.x × blockDim.x = blocks per grid × threads per block
```

> Launch geometry is part of the algorithm: it defines which threads exist.

---

## Hardware Schedules Blocks onto SMs

![h:300](img/w08-thread-block-scheduling.png)

- A block executes on **one SM**.
- An SM can have multiple **active blocks**, resources permitting.
- The grid can contain far more blocks than the GPU has SMs.
- Blocks are scheduled in **no guaranteed order**.

<!-- _footer: "Source: NVIDIA CUDA Programming Guide, Fig. 4." -->

---

## The Scalability Contract

The same program can run on small and large GPUs because you do **not** specify:

- which SM receives a block;
- how many blocks are resident at once;
- the order in which blocks run.

Therefore:

1. Do not make one block depend on another block's intermediate result.
2. Do not assume block 0 finishes before block 1 starts.
3. Use separate kernel launches when you need a grid-wide completion boundary.

> Within a block, threads can cooperate. Across blocks, assume independence.

---

## Warps and SIMT: The Execution Granularity

![h:245](img/w08-active-warp-lanes.png)

- A **warp** contains 32 threads scheduled together on an SM.
- Under **SIMT**, one instruction stream is applied across thread lanes.
- If threads take different branches, subsets are active at different times: **divergence**.
- Modern GPUs support independent thread scheduling; do not write code that relies on implicit lockstep.

*Source: NVIDIA CUDA Programming Guide, Fig. 7 (even-indexed lanes active).*

---

## Divergence: Correctness vs. Performance

```python
@cuda.jit
def branchy(x, out):
    i = cuda.grid(1)
    if i < x.size:
        if x[i] >= 0:
            out[i] = x[i]       # one subset of a warp
        else:
            out[i] = -x[i]      # another subset
```

- The code is correct: each thread follows its own path.
- A warp may spend time executing both paths with different active lanes.
- Occasional boundary checks are fine; highly irregular work can reduce efficiency.

> First make the mapping correct. Profile before redesigning around divergence.

---

## Memory in the Programming Model

| Storage | Visible to | Lifetime / role | This week |
|---|---|---|---|
| Registers / local variables | One thread | Intermediate values | Used implicitly |
| Shared memory | Threads in one block | Cooperation and reuse | Preview; Week 9 |
| Global device memory | All threads in the grid | Inputs, outputs, device arrays | Used explicitly |
| Host memory | CPU | NumPy arrays and application data | Copied to/from device |

**Important:** “local memory” is per-thread, but is not necessarily fast or on-chip.
“Shared memory” is on-chip scratchpad shared by a block.

> Correct scope and ownership first; memory optimisation next week.

---

## Two Different Synchronization Levels

| Mechanism | Scope | Purpose |
|---|---|---|
| `cuda.syncthreads()` | Threads in **one block** | Barrier before/after shared-memory cooperation |
| `cuda.synchronize()` | Host waits for device work | Ensure completion before timing or using results |

Rules:

- Every thread in a block must reach the same `cuda.syncthreads()` barrier.
- Do not place that barrier inside a condition that only some threads enter.
- Kernel launches are asynchronous with respect to the host.
- Operations in one stream are ordered; streams and overlap come later.

---

## The Same Model in CUDA C++ and Numba CUDA

| General CUDA concept | CUDA C++ spelling | Numba CUDA spelling |
|---|---|---|
| Kernel declaration | `__global__ void k(...)` | `@cuda.jit def k(...):` |
| Thread in block | `threadIdx.x` | `cuda.threadIdx.x` |
| Block in grid | `blockIdx.x` | `cuda.blockIdx.x` |
| Threads in block | `blockDim.x` | `cuda.blockDim.x` |
| Launch | `k<<<blocks, threads>>>(...)` | `k[blocks, threads](...)` |
| Block barrier | `__syncthreads()` | `cuda.syncthreads()` |
| Device synchronisation | `cudaDeviceSynchronize()` | `cuda.synchronize()` |

> The language changes. The execution, memory, and synchronization model does not.

---

## Thread Indexing: The Core Formula

```python
# Each thread computes its global index:
i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
```

| Symbol | Meaning | Example |
|--------|---------|---------|
| `blockIdx.x` | Which block am I in? | Block 3 |
| `blockDim.x` | How many threads per block? | 256 |
| `threadIdx.x` | Which thread within my block? | Thread 42 |
| **Global index** | `3 * 256 + 42 = 810` | Element 810 |

**Worked example:** `N = 1,000,000`, 256 threads/block →
`3,907 blocks`, `1,000,192 launched threads`, and `192` threads exit.

> Write what thread `i` does—not the host-side `for i in range(N)` loop.
> Hardware schedules the launched threads in warps and waves.

---

## 2D Indexing: For Matrices

Use **x for columns** and **y for rows** so consecutive x-threads read adjacent columns.

```python
col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

# Equivalent shortcut:
# col, row = cuda.grid(2)

if row < n_rows and col < n_cols:
    out[row, col] = ...
```

```python
threads = (16, 16)  # (x=columns, y=rows), 256 threads
blocks = ((n_cols + 15) // 16,
          (n_rows + 15) // 16)
kernel[blocks, threads](...)
```

> `cuda.grid(2)` returns `(x, y)`, **not automatically `(row, col)`**.
> For a 4096×4096 matrix, this launch creates 256×256 blocks and 16.7M threads.

---

## Bounds Checking: Don't Read Past the Array

```python
i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
if i < N:                    # ← CRITICAL
    c[i] = a[i] + b[i]
```

Why? Grid size is rounded up:
- N = 1,000,000, blockDim = 256
- Grid = ⌈1,000,000 / 256⌉ = 3,907 blocks
- Total threads launched = 3,907 × 256 = 1,000,192
- **192 threads have no work.** Without the check → out-of-bounds access → crash or garbage.

> Always include the bounds check. It's nearly free (one comparison per thread).

---

## Launch Configuration: Choosing Block and Grid Size

| Parameter | Typical choice | Constraint |
|-----------|---------------|------------|
| `blockDim.x` (threads per block) | Start with 128 or 256 | Max 1024 threads/block; a multiple of 32 is a performance recommendation, not a syntax rule. |
| Grid size | `⌈N / blockDim⌉` | Can be very large (millions) |

```python
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
my_kernel[blocks_per_grid, threads_per_block](a, b, c, N)
```

> **Starting point, not a law:** try 128 or 256 for simple element-wise kernels,
> then measure. Week 9 covers occupancy and tuning.

---

## Optional Pattern: The Grid-Stride Loop

One thread per element is simple, but not the only valid mapping.

```python
i = cuda.grid(1)          # first element for this thread
stride = cuda.gridsize(1) # total threads in the launch

while i < c.size:
    c[i] = a[i] + b[i]
    i += stride
```

- Each thread handles several elements, `stride` positions apart.
- A fixed-size grid can process any `N` without launching one thread per element.
- Consecutive threads still access consecutive addresses in each iteration.

> Use one-thread-per-element for today's first kernels; remember grid-stride
> for larger or reusable kernels. Correctness still requires the `while i < N` check.

---

<!-- _class: lead -->

# Part III
# Numba CUDA: Your First Kernel

---

## What Is Numba CUDA?

> Numba CUDA compiles a **restricted subset of Python** into device code
> that follows the CUDA execution model.

It is not a way to run arbitrary Python objects or libraries inside a kernel.

```python
from numba import cuda

@cuda.jit
def my_kernel(a, b, c):
    i = cuda.grid(1)           # global thread index (1D)
    if i < c.size:
        c[i] = a[i] + b[i]
```

| Feature | CuPy | Numba CUDA |
|---------|------|-----------|
| Abstraction level | Array operations | Individual threads |
| You control | What to compute | How each thread computes |
| Use when | Standard operations | Custom / fused operations |

---

## Environment Setup: Numba CUDA

**Installation (choose the toolkit matching your environment):**

```bash
pip install "numba-cuda[cu12]"   # or: pip install "numba-cuda[cu13]"
```

<!-- Instructor ecosystem note, checked November 2026:
The Numba-CUDA documentation describes the project as maintenance mode through
CUDA 13; new feature development targets Numba-CUDA-MLIR. Keep this week's focus
on the stable SIMT concepts, which transfer to CUDA C++, Triton, and newer APIs.
Source: https://nvidia.github.io/numba-cuda/
-->

---

## The `@cuda.jit` Decorator

```python
from numba import cuda
import numpy as np

@cuda.jit
def vector_add(a, b, c):
    """Each thread adds one element."""
    i = cuda.grid(1)              # this thread's global index
    if i < c.size:                # bounds check
        c[i] = a[i] + b[i]
```

**Key Numba CUDA functions:**

| Function | Meaning |
|----------|---------|
| `cuda.grid(1)` | Global index in 1D (= `blockIdx.x * blockDim.x + threadIdx.x`) |
| `cuda.grid(2)` | `(x, y)` in 2D; this lecture maps `x→col`, `y→row` |
> `cuda.blockIdx`, `cuda.threadIdx`, `cuda.blockDim`, and `cuda.gridDim`
> expose the lower-level geometry from Part II. The first launch triggers JIT compilation.

---

## Launching a Kernel

```python
# 1. Allocate arrays on GPU
N = 1_000_000
a = cuda.to_device(np.random.rand(N).astype(np.float32))
b = cuda.to_device(np.random.rand(N).astype(np.float32))
c = cuda.device_array(N, dtype=np.float32)     # allocate on GPU, uninitialised

# 2. Configure launch
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# 3. Launch: kernel[grid, block](args)
vector_add[blocks_per_grid, threads_per_block](a, b, c)

# 4. Wait for completion and copy back
cuda.synchronize()
result = c.copy_to_host()
```

> The `[grid, block]` syntax is Numba's way of specifying launch configuration.
> The kernel is **asynchronous**: Python continues immediately after launch.
> `cuda.synchronize()` blocks until all GPU work is done.

---

## Full Working Example: Vector Addition

> 📁 `class-examples/week08/01_vector_add.py`

```python
import numpy as np
from numba import cuda

@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

N = 10_000_000
a_np = np.random.rand(N).astype(np.float32)
b_np = np.random.rand(N).astype(np.float32)
a_d, b_d = cuda.to_device(a_np), cuda.to_device(b_np)
c_d = cuda.device_array(N, dtype=np.float32)  # uninitialised output

tpb = 256
bpg = (N + tpb - 1) // tpb
vector_add[bpg, tpb](a_d, b_d, c_d)  # asynchronous launch
c_np = c_d.copy_to_host()            # waits for the result
assert np.allclose(c_np, a_np + b_np)
```

---

## Live Demo: Run It on Colab

```python
# First: verify GPU is available
!nvidia-smi

# Check Numba sees the GPU
from numba import cuda
print(cuda.is_available())        # True
print(cuda.get_current_device())  # Tesla T4
```

> Run `01_vector_add.py` live. Ask students:
> - "How many threads are launched?" → 39,063 × 256 = **10,000,128**
> - "How many blocks?" → **39,063**
> - "What does each thread do?" → One addition
> - "Where is the parallelism?" → In the launch, not in a loop

---

## Kernel Anatomy: Dissecting the Code

```python
@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)          # ① "Who am I?"
    if i < c.size:            # ② "Is there work for me?"
        c[i] = a[i] + b[i]   # ③ "Do my one piece of work"
```

| Step | What happens | Parallelism |
|------|-------------|-------------|
| ① | Each thread computes its unique index | Across the launched grid |
| ② | Threads beyond array size exit | 128 threads in the 10M example |
| ③ | Each in-bounds thread does ONE addition | In warps, blocks, and waves |

> **The mental shift:** you don't write `for i in range(N)`.
> You write what thread `i` does. The launch supplies the index space; hardware schedules it.

---

## Common Mistakes: Indexing, Launches, and Transfers

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forgetting bounds check | Illegal access, garbage, or later CUDA error | Check `i < N` when the launch is rounded up |
| Wrong grid size | Some elements uncomputed | Use `ceil(N / tpb)`, not floor division |
| Keeping an `N`-element loop in every thread | Duplicated work and possible races | Map indices to work; use per-thread loops intentionally |
| Passing host arrays implicitly | Hidden host↔device copies and synchronous behaviour | Use `cuda.to_device()` explicitly |
| Forgetting synchronization | Timing measures launch, not completion | Synchronize before reading elapsed time |

---

## Common Mistakes: Races, Barriers, and Debugging

| Mistake | Why it is wrong | Fix |
|---------|-----------------|-----|
| Many threads write one location | Race; result can depend on order | Use a reduction or an appropriate atomic |
| Early return before a block barrier | Not all block threads reach the barrier | Restructure so every thread reaches `syncthreads()` |
| Excessive kernel `print()` | Noisy output and timing distortion | Use small debug launches; avoid prints in benchmarks |
| Optimising before verification | Fast code may compute the wrong answer | Compare against a CPU reference first |

> Debugging order: **tiny input → correct result → synchronize → benchmark → profile**.

---

<!-- _class: lead -->

# Part IV
# Memory Transfers and Device Arrays

---

## The Two Memory Spaces (Recap from Week 6)

```text
┌─────────────────────┐         ┌─────────────────────┐
│   CPU (Host)        │         │   GPU (Device)      │
│                     │  PCIe   │                     │
│   numpy.ndarray     │◄──────►│   cuda device array │
│   System RAM        │         │   Device DRAM      │
│                     │         │                     │
└─────────────────────┘         └─────────────────────┘
```

| Direction | Numba CUDA function | Cost |
|-----------|-------------------|------|
| Host → Device | `cuda.to_device(np_array)` | Allocation + copy; measure on your system |
| Device → Host | `device_array.copy_to_host()` | Synchronous by default without a stream |
| Allocate on GPU | `cuda.device_array(shape, dtype)` | Fast (no transfer) |

> **Rule from Week 6:** Upload once, compute a lot, download once.
> Avoid **unnecessary** transfers inside hot loops; interconnect bandwidth is system-dependent.

---

## Device Arrays in Numba CUDA

```python
import numpy as np
from numba import cuda

x_np = np.random.rand(1000).astype(np.float32)
x_d = cuda.to_device(x_np)                 # allocate + copy host→device
y_d = cuda.device_array(1000, dtype=np.float32)  # empty device array
z_d = cuda.device_array_like(x_d)          # empty; same shape/dtype

x_back = x_d.copy_to_host()                # copy device→host
assert np.allclose(x_np, x_back)
```

| Operation | Does it copy data? | Initialised? |
|---|:---:|:---:|
| `cuda.to_device(x_np)` | Yes, host→device | Yes, from `x_np` |
| `cuda.device_array(...)` / `device_array_like(...)` | No | **No** — like `np.empty` |
| `.copy_to_host()` | Yes, device→host | Yes, returned host array |

> Explicit transfers make costs visible and avoid accidental synchronous copies.

---

## The Transfer Cost: Think End-to-End

```text
T_total = T_upload + T_kernel + T_download
T_GPU_speedup_is_useful_only_if T_total < T_CPU
```

| Regime | Dominant issue | Design response |
|---|---|---|
| Tiny array | Launch and transfer latency | Keep work on CPU |
| Large array, trivial kernel | Data movement and memory bandwidth | Fuse operations; avoid transfers |
| Large array, substantial compute | GPU computation can amortise overhead | Reuse device data across launches |

For float32 inputs `a`, `b` and output `c`, vector addition moves at least
**3 × N × 4 bytes** inside device memory, in addition to any host↔device copies.

> Do not reuse a universal “crossover size.” It depends on the CPU, GPU,
> interconnect, dtype, allocation strategy, and whether data is already resident.
> Measure compute-only **and** end-to-end time, as in Week 6.

---

## Minimising Transfers: The Pattern

```python
# ❌ BAD when the CPU modification is unnecessary: transfer every iteration
for epoch in range(100):
    x_d = cuda.to_device(x_np)        # upload every time!
    my_kernel[bpg, tpb](x_d, y_d)
    y_np = y_d.copy_to_host()          # download every time!
    # ... modify y_np on CPU ...

# ✅ GOOD: upload once, compute many times, download once
x_d = cuda.to_device(x_np)             # upload ONCE
for epoch in range(100):
    my_kernel[bpg, tpb](x_d, y_d)     # stay on GPU
cuda.synchronize()
y_np = y_d.copy_to_host()              # download ONCE
```

> Same principle as Week 5's object store: put data once, use many times.
> Same principle as Week 6's CuPy: intermediates stay in VRAM.

---

## Synchronization: When Is the GPU Done?

```python
# Kernel launches are ASYNCHRONOUS:
my_kernel[bpg, tpb](a_d, b_d, c_d)   # returns immediately!
# GPU might still be computing...

cuda.synchronize()                     # NOW wait for all GPU work to finish
result = c_d.copy_to_host()            # safe to read
```

| Function | What it does |
|----------|-------------|
| `cuda.synchronize()` | Block until ALL GPU work is done |
| `.copy_to_host()` | Implicitly synchronises (must wait for data) |
| `cuda.stream` | Advanced: overlap transfers with compute (Week 9) |

> **Timing rule (from Week 6):** Always `cuda.synchronize()` before reading the clock.
> Without it, wall-clock timing may measure only the launch, not the kernel.

---

<!-- _class: lead -->

# Part V
# AI Examples: From ReLU to Attention

---

## Example 1: Element-wise ReLU

```python
@cuda.jit
def relu_kernel(x, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = max(0.0, x[i])
```

- One thread per element; each performs one comparison and at most one write.
- A 256×512 batch-hidden matrix launches **131,072 threads**, scheduled in waves.

```python
out_cupy = cp.maximum(x_gpu, 0)  # same operation using a library kernel
```

> For standard ReLU, use CuPy/PyTorch. This example exposes the mapping inside it.

---

## Example 2: Fused ReLU + Bias

```python
@cuda.jit
def fused_relu_bias(x, bias, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = max(0.0, x[i]) + bias[i % bias.size]
```

```text
Unfused: relu writes a full temporary; bias_add reads it again
Fused:   no full-size temporary; one launch writes the final output
```

- Fewer launches and less global-memory traffic; the bias is read in both versions.
- Actual speedup depends on size and hardware: **measure**.

> `03_fused_relu_bias.py` benchmarks both versions and verifies the result.

---

## Example 3: Row-wise Softmax Needs Cooperation

```text
For each row r:
  m = max_c logits[r, c]
  s = Σ_c exp(logits[r, c] - m)
  out[r, c] = exp(logits[r, c] - m) / s
```

A natural mapping is **one block per row, one thread per column**, but:

- every output needs the row maximum and row sum;
- those are **reductions**, not independent element-wise operations;
- an efficient implementation needs shared memory and block barriers.

> `04_softmax_kernel.py` is deliberately naïve: each thread recomputes the same
> row maximum and sum. It is correct under its small-row assumptions, but slow.
> Week 9 introduces cooperative reductions.

---

## Example 4: Attention Scores (The AI Workload)

```python
@cuda.jit
def attention_scores(Q, K, scores):
    col, row = cuda.grid(2)  # x=key, y=query
    if row < Q.shape[0] and col < K.shape[0]:
        dot = 0.0
        for d in range(Q.shape[1]):
            dot += Q[row, d] * K[col, d]
        scores[row, col] = dot
```

```python
threads = (16, 16)
blocks = ((K.shape[0] + 15) // 16,
          (Q.shape[0] + 15) // 16)
attention_scores[blocks, threads](Q, K, scores)
```

- Each thread computes **one score**; its per-thread loop reduces over `d`.
- Sequence length 512 → **262,144 outputs**.
- This naïve kernel has no tiling or data reuse. Use cuBLAS in practice; Week 9 introduces tiles.

---

## Comparing Your Kernel vs. CuPy

| Operation | Reasonable expectation | Main reason |
|-----------|------------------------|-------------|
| Vector add / ReLU | Can approach library performance | Simple mapping; usually bandwidth-bound |
| Naïve matmul | Usually much slower than cuBLAS | No tiling, reuse, or architecture tuning |
| Naïve softmax | Usually slower than optimised libraries | Reductions and synchronisation matter |
| Custom fused op | **May outperform separate library calls** | Less traffic and fewer intermediates |

> These are hypotheses, not universal benchmark numbers. Measure on your GPU.
>
> **Rule:** use libraries for standard ops; write kernels for custom operations,
> fusion, or learning. A first kernel does not need to beat cuBLAS to be useful.

---

<!-- _class: lead -->

# Part VI
# Performance: When Your Kernel Wins (and Loses)

---

## Why Your First Kernel Is Slow (and That's OK)

Your vector_add kernel:
```python
c[i] = a[i] + b[i]    # 2 reads + 1 write per thread
```

Memory traffic per element: 4 + 4 + 4 = 12 bytes
FLOPs per element: 1

Arithmetic intensity = 1/12 ≈ 0.08 FLOP/byte → **extremely bandwidth-bound**

> On T4 (320 GB/s): theoretical max = 320 × 0.08 ≈ 27 GFLOP/s
> T4 FP32 peak is about 8 TFLOP/s, so this roofline is roughly 0.3% of peak.
> That is a model of the bottleneck, not a measured utilisation figure.
> The bottleneck is memory bandwidth, not arithmetic.

> This is the roofline model from Week 6, applied to your kernel.

---

## Occupancy: Keeping the GPU Busy

> The GPU hides memory latency by running **many warps simultaneously**.
> If too few threads are active, the GPU idles while waiting for memory.

| Factor | Effect on occupancy |
|--------|-------------------|
| Blocks too small for the workload | May underfill an SM and hit per-block limits |
| Registers per thread too high | Fewer threads fit per SM |
| Shared memory per block too high | Fewer blocks fit per SM |

> **For today:** 128 or 256 threads/block is a reasonable starting point, not a universal optimum.
**Week 9:** we'll use the CUDA occupancy calculator and tune.

---

## Memory Coalescing (Preview of Week 9)

```text
Good:  Thread 0 reads a[0], Thread 1 reads a[1], Thread 2 reads a[2], ...
       → A warp touches a contiguous region; hardware needs few transactions.

Bad:   Thread 0 reads a[0], Thread 1 reads a[1000], Thread 2 reads a[2000], ...
       → A warp touches many separate regions; more memory transactions.
```

> Your vector_add and ReLU kernels have **coalesced-friendly** access:
> consecutive threads read consecutive elements.
>
> The main problem in naïve attention/matmul is **lack of data reuse**: many
> threads reread the same row/column data from global memory. Week 9 uses shared
> memory tiling to reduce that traffic; coalescing and tiling are related but distinct.

---

## Benchmarking Your Kernel

```python
from numba import cuda

# Warm up (first call includes JIT compilation)
vector_add[bpg, tpb](a_d, b_d, c_d)
cuda.synchronize()

# Timed runs
start = cuda.event()
end = cuda.event()
start.record()
for _ in range(10):
    vector_add[bpg, tpb](a_d, b_d, c_d)
end.record()
end.synchronize()
elapsed = cuda.event_elapsed_time(start, end) / 10  # ms per call
```

> Use CUDA events for device-side timing. Warm up the JIT, repeat runs,
> report a mean or median, and inspect variability.

<!-- _footer: "Example: class-examples/week08/05_benchmark_kernels.py" -->

---

<!-- _class: lead -->

# Part VII
# Connection to the Course + Lab

---

## The Four Questions on GPU (Updated)

| Question | Week 6 (CuPy) | This week (Numba CUDA) |
|----------|--------------|----------------------|
| **Split** | CuPy splits automatically | You choose: one thread per element |
| **Assign** | Hardware assigns blocks to SMs | Same (you don't control) |
| **Move** | `cp.asarray` / `.get()` | `cuda.to_device` / `.copy_to_host()` |
| **Wait** | Implicit in `.get()` | `cuda.synchronize()` |

> The questions don't change. The level of control does.
> CuPy: you say WHAT. Numba CUDA: you say HOW (per thread).

---

## Where This Connects Forward

| This week | Where it reappears |
|-----------|-------------------|
| Thread indexing | Every CUDA kernel you'll ever read (Week 9, papers) |
| Bounds checking | `torch.compile` generated code (Week 13) |
| Kernel fusion | `torch.compile` does it automatically (Week 13) |
| Memory transfers | DataLoader → GPU (Week 10), DDP gradient sync (Week 11) |
| Coalescing | Shared memory tiling for matmul (Week 9) |
| Occupancy | Kernel resource and launch-configuration tuning (Week 9) |

> You now understand what happens inside `torch.matmul`, `F.relu`,
> and every CUDA kernel that PyTorch launches. Week 9 goes deeper.

---

## Key Takeaways: Model and Correctness

1. **One thread's code; a grid of work.** A launch supplies the index space.
   Per-thread loops remain valid for reductions and grid-stride mappings.

2. **Indexing:** `i = cuda.grid(1)` in 1D; `(col, row) = cuda.grid(2)` here.
   Always bounds-check when the launch is rounded up.

3. **Launch:** `kernel[blocks, threads](args)`; launched blocks × threads ≥ `N`.

4. **Scheduling is not a guarantee.** Blocks run in any order.
   `cuda.syncthreads()` synchronises only within one block.

---

## Key Takeaways: Data and Performance

5. **Memory:** upload once, reuse device data, download once.
   Judge transfers end-to-end; avoid accidental host-array copies.

6. **Synchronize** before timing or reading results; use CUDA events for kernel time.

7. **Libraries first.** Write kernels for custom or fused operations.
   Do not expect a naïve kernel to beat cuBLAS.

8. **Fusion can reduce memory traffic** and intermediate allocations.
   Verify correctness, then measure; do not assume a fixed speedup.

---

## Lab: Four Exercises on Colab

> 📁 `class-examples/week08/lab_week08_kernels.ipynb`

| # | Exercise | What you learn |
|---|----------|---------------|
| 1 | Vector addition (10M elements) | Thread indexing, launch config, transfers |
| 2 | Element-wise ReLU on a 256×512 matrix | Same pattern, AI-shaped data |
| 3 | Fused ReLU + bias add (one kernel vs two) | Fusion benefit: measure the speedup |
| 4 | Your kernel vs CuPy for vector add | When library wins, when yours matches |

**Deliverable:** Results table + one sentence:
"My kernel beats CuPy when... and loses when..."

**Bonus:** Implement a 2D kernel for matrix transpose. Compare against `cp.transpose`.

---

## Pre-Class Checklist (for you)

- [ ] Verify Numba CUDA works on Colab:
  ```python
  from numba import cuda
  print(cuda.is_available())  # True
  ```
- [ ] Run all 5 class-example scripts on Colab T4, record timings
- [ ] Prepare the Colab notebook with cells pre-written (students fill in kernels)
- [ ] Test `cuda.event_elapsed_time` for timing (more reliable than perf_counter)
- [ ] Have a CPU fallback ready in case Colab GPU quota is exhausted

---

## Official References

- [NVIDIA CUDA Programming Guide — Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [Numba CUDA — Writing Kernels](https://nvidia.github.io/numba-cuda/user/kernels.html) · [Memory](https://nvidia.github.io/numba-cuda/user/memory.html) · [Installation](https://nvidia.github.io/numba-cuda/user/installation.html)
- [Numba CUDA project status](https://nvidia.github.io/numba-cuda/) · [Course repository](https://github.com/adaskin/hpc-for-ai-course)

**Official figures used:** NVIDIA Programming Guide Figures 2, 3, 4, and 7.
Local copies and full credits are bundled under `slides/img/` and in `08cuda-kernel.md`.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank You

### Next week: CUDA Memory Hierarchy & Profiling (Week 9)

*YZM 513 – İstanbul Medeniyet Üniversitesi – Fall 2026*
