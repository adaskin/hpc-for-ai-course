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
  footer {
    font-size: 16px;
    opacity: 0.5;
  }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# YZM 513 – High-Performance Programming for AI

### Week 3: Why Parallelism, Race Conditions & Synchronisation

**İstanbul Medeniyet Üniversitesi**
AI Engineering M.Sc. – Fall 2026

Ammar Daşkın

---

## Recap & Today's Arc

**Weeks 1–2:** We made *single-threaded* code fast (vectorise, JIT).

**Today's question:** What if one core isn't enough?

**Today's arc:**

```text
WHY parallelism?  →  HOW threads work  →  WHAT goes wrong?
(performance)        (mechanics)           (race conditions)
                                                │
                                                ▼
                                    HOW to fix it?
                                    (synchronisation)
                                                │
                                                ▼
                                    Python's twist: the GIL
```

> We'll use **C examples** first (clearer, no GIL confusion),
> then show **Python/NumPy equivalents**.

---

<!-- _class: lead -->

# Part I
# Why Do We Need Parallel Computation?

---

## The Performance Problem

Last week: vectorisation gave ~1000× speedup *on one core*.

But many AI workloads can't be expressed as a single array operation:
- Loading and augmenting **thousands of images** from disk
- Running **independent experiments** (hyperparameter search)
- Processing a **batch** where each sample is independent
- Computing **gradients** across data shards

> One core doing these sequentially wastes the other 7–15 cores
> sitting idle on your CPU.

**Parallelism = using multiple cores to reduce wall-clock time.**

---

## Serial vs. Parallel: A Concrete Example

### Summing an array of 100 million elements

```text
Serial (1 core):
  Core 0: [████████████████████████████████████████]
           |←──────────── 4.0 seconds ────────────→|

Parallel (4 cores, each sums 25M elements):
  Core 0: [██████████]  ← sum[0:25M]
  Core 1: [██████████]  ← sum[25M:50M]
  Core 2: [██████████]  ← sum[50M:75M]
  Core 3: [██████████]  ← sum[75M:100M]
           |←─ 1.0 s ─→|
           + tiny reduction step to combine 4 partial sums
```

> **~4× speedup** (minus small overhead for combining results).

<!-- note:
This is data parallelism: same operation (sum), different data chunks.
The reduction step (combining 4 partial sums) is the serial part.
Amdahl's Law: if the reduction takes 1ms out of 1000ms total,
the serial fraction is 0.1% → max speedup ≈ 1000×. Practically: ~4×.
-->

---

## Data Parallelism: Matrix Addition

From the operation $T = A + B$ where $A, B \in \mathbb{R}^{m \times n}$:

$$t_{ij} = a_{ij} + b_{ij} \quad \forall\; i, j$$

```text
┌─────────────────────────────────────────────────────────┐
│  Each element t[i,j] depends ONLY on a[i,j] and b[i,j] │
│  → All m×n additions are INDEPENDENT                    │
│  → Perfect data parallelism                             │
└─────────────────────────────────────────────────────────┘
```

| Cores | Work per core | Time (relative) |
|:-----:|:-------------:|:---------------:|
| 1 | $m \times n$ additions | $T$ |
| 4 | $m \times n / 4$ each | $T/4$ |
| 8 | $m \times n / 8$ each | $T/8$ |

> **In AI:** element-wise activations (ReLU, sigmoid), batch normalisation,
> gradient scaling — all data-parallel.

---

## Data Parallelism: Matrix Multiplication

$C = A \cdot B$ where $A \in \mathbb{R}^{m \times p}$, $B \in \mathbb{R}^{p \times n}$:

$$c_{ij} = \sum_{k=1}^{p} a_{ik} \cdot b_{kj}$$

- Each $c_{ij}$ is **independent** of other elements of $C$.
- But computing one $c_{ij}$ requires a **reduction** (sum over $k$).

```text
Row i of A  ×  Column j of B  →  c[i,j]

  [a_i1, a_i2, ..., a_ip]  ·  [b_1j, b_2j, ..., b_pj]^T
       ↓                            ↓
  Parallel across (i,j) pairs    Serial within each dot product
                                  (but can use SIMD / tree reduction)
```

> **In AI:** every linear layer, attention score computation (Q@K^T).
> BLAS libraries parallelise this internally across cores.

---

## Task Parallelism: Different Operations, Same Time

Given matrices $A$ and $B$, compute **both**:
- $T = A + B$ (addition)
- $C = A \cdot B$ (multiplication)

These are **independent tasks** on the **same data**:

```text
┌────────────────────────────────────────────────────┐
│                                                    │
│  Core group 1:  T = A + B   (addition task)       │
│                                                    │
│  Core group 2:  C = A × B   (multiplication task) │
│                                                    │
│  Both run SIMULTANEOUSLY. No dependency.           │
└────────────────────────────────────────────────────┘
```

> **In AI:** data loading (task 1) + augmentation (task 2) +
> training (task 3) + logging (task 4) — all running concurrently.

---

## Data vs. Task Parallelism: Summary

| | Data Parallelism | Task Parallelism |
|---|---|---|
| **Split** | Data into chunks | Work into different tasks |
| **Operation** | Same op, different data | Different ops |
| **Granularity** | Fine (per element) | Coarse (per task) |
| **Communication** | Often needed (reduction) | Minimal |
| **AI example** | Batch through a layer | Load + preprocess + train |
| **Hardware** | GPU (SIMT), SIMD, BLAS | Multi-core CPU, cluster |

> Real AI systems use **both simultaneously**:
> data-parallel within a GPU kernel, task-parallel across pipeline stages.

---

## Embarrassingly Parallel Problems

> A problem is **embarrassingly parallel** if it requires minimal effort
> to split into independent tasks.

Examples:
- Apply `f(x)` to every element of an array → `map(f, arr)`
- Process each image in a dataset independently
- Run 100 independent training experiments
- Monte Carlo simulation: each sample is independent

```c
// C: parallel map
for (int i = 0; i < N; i++)
    result[i] = func(arr[i]);  // each iteration independent
```

> **No synchronisation needed.** Each output depends only on its own input.
> This is the "easy" case. Today we'll also see the "hard" case.

---

## Amdahl's Law (Recap from Week 1)

$$S(N) = \frac{1}{(1 - P) + \dfrac{P}{N}}$$

| Parallel fraction $P$ | $N=4$ | $N=8$ | $N \to \infty$ |
|:---:|:---:|:---:|:---:|
| 0.90 | 2.1× | 2.4× | **10×** |
| 0.95 | 2.4× | 2.9× | **20×** |
| 0.99 | 3.1× | 3.7× | **100×** |

> The **serial fraction** (1−P) is your ceiling.
> Synchronisation (locks, barriers) **adds** to the serial fraction.
> Minimise time spent in synchronised sections.

---

<!-- _class: lead -->

# Part II
# Implementing Parallelism: Let's Make It Fast

---

## The Pattern: Split → Distribute → Combine

Any parallel computation follows three steps:

```text
┌─────────────────────────────────────────────────────────────────┐
│  1. SPLIT       Divide data/work into independent chunks        │
│  2. DISTRIBUTE  Give each chunk to a worker (thread/process)    │
│  3. COMBINE     Collect partial results → final answer          │
└─────────────────────────────────────────────────────────────────┘
```

```text
Data: [████████████████████████████████████████]
       ↓ SPLIT
       [chunk 0] [chunk 1] [chunk 2] [chunk 3]
       ↓ DISTRIBUTE
       Worker 0   Worker 1   Worker 2   Worker 3
       ↓ COMBINE
       [partial₀] [partial₁] [partial₂] [partial₃] → final result
```

> **Today's task:** compute-heavy transform on 20 million numbers.
> Embarrassingly parallel: each chunk is independent.

---

## Two Ways to Create Workers

| | **Threads** | **Processes** |
|---|---|---|
| Share memory? | ✅ Yes (same heap) | ❌ No (separate address space) |
| Creation cost | Light (~µs) | Heavier (~ms) |
| Communication | Direct (shared variables) | Must **serialise** (pickle) |
| Python module | `threading` / `ThreadPoolExecutor` | `multiprocessing` / `ProcessPoolExecutor` |
| GIL? | Shared (one GIL) | Each has own GIL |

> For today we'll use `concurrent.futures` which gives us
> both with the **same API**. One line change: Thread → Process.

---

## The Hidden Cost: Pickling

Every argument you pass to a `ProcessPoolExecutor` worker must be
**serialised (pickled)** and sent over an OS pipe. The result comes back the same way.

```text
Parent process                                Child process
──────────────                                ─────────────
data (160 MB, resident)
  │
  │  pickle.dumps(chunk_0)   ← 20 MB array → ~20 MB bytes
  │  write to pipe ─────────────────────────►  unpickle chunk_0
  │                                              compute
  │  pickle.dumps(chunk_1)   ← serial, blocks parent
  │  write to pipe ─────────────────────────►  unpickle chunk_1
  │                                              compute
  │  ...
  │  read results  ◄──────────────────────────  pickle.dumps(result)
  ▼
combine
```

> **Key point:** the parent pickles serially. Even with 8 workers alive,
> only one chunk is being shipped at a time. Workers sit idle waiting.

<!-- note:
This is the single most common mistake with multiprocessing.
see `ex.map(f, chunks)` and think the array is "shared". It is NOT.
Every argument crosses an IPC boundary.
Rule: if the argument is bigger than a few KB, you have a problem.
-->

---

## Live Demo: What Naive Pickling Looks Like

The "obvious" implementation — pass chunks to workers:

```python
# ❌ NAIVE: pickles 160 MB per run
def partial_sum_of_squares(chunk):
    return np.sum(chunk ** 2)

chunks = np.array_split(data, n_workers)     # 20M floats → 160 MB
partials = list(ex.map(partial_sum_of_squares, chunks))
```

> 🛠 **Run:** `python parallel_sum_with_pickling_overhead.py`

```text
Workers   Time (s)    Speedup   Efficiency
──────────────────────────────────────────────
serial    0.365       1.00×     100%
2         2.243       0.16×       8%
4         2.747       0.13×       3%
8         3.078       0.12×       1%
```

> **Adding workers made it 8× SLOWER.** The compute is 0.36 s.
> The pickling + IPC is ~2.4 s. The workers never had a chance.

<!-- note:
"What do you think is happening?"
It's not GIL (processes have no GIL). It's not memory contention.
The parent is spending all its time in pickle.dumps, not in np.sum.
You can prove it with cProfile:
  python -m cProfile -s cumtime parallel_sum_naive.py
You'll see pickle.dumps near the top.
-->

---

## The Fix: Pass Indices, Not Data

```python
# ❌ BAD: pickles the data (160 MB per run)
chunks = np.array_split(data, n_workers)
ex.map(partial_sum_of_squares, chunks)     # 8 × 20 MB pickled

# ✅ GOOD: pickles only (start, end) — a few bytes each
data = np.random.rand(SIZE)                 # module-level global

def partial_sum(bounds):
    start, end = bounds
    return float(np.sum(transform(data[start:end])))

bounds = [(i * SIZE // n, (i+1) * SIZE // n) for i in range(n)]
ex.map(partial_sum, bounds)                 # 8 × ~16 bytes pickled
```

> How is `data` visible inside the worker? The child process
> **inherits the parent's memory** via `fork()` (copy-on-write on Linux).
> Reading `data[start:end]` touches shared pages — no pickle, no copy.

```text
Parent memory                  Child memory (fork)
─────────────                  ───────────────────
data: [████ 160 MB ████] ──COW─► data: (same physical pages, read-only)
                                  └─ child reads data[s:e] directly
```

---

## Fork vs. Spawn: Which Start Method?

| Start method | Default on | How children get data |
|---|---|---|
| `fork` | Linux | Inherit parent memory (copy-on-write) — **free** |
| `spawn` | macOS, Windows | Fresh interpreter — **everything pickled** |
| `forkserver` | (opt-in) | Fork from clean server — safer with threads |

```python
import multiprocessing as mp

# Explicitly choose fork (Linux) so children inherit `data` for free:
ctx = mp.get_context("fork")
with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
    partials = list(ex.map(partial_sum, bounds))
```

> ⚠️ **Portability:** `fork` is Linux-only. On macOS/Windows the demo
> would fall back to `spawn` and slow down again.
> Cross-platform fix: use `multiprocessing.shared_memory`.

---

## Rule of Thumb: Three Ways Out

If a worker argument is bigger than a few kilobytes,
you have a pickling problem.

| Approach | How it works | When to use |
|---|---|---|
| **Pass indices / bounds** | Worker reads inherited or memory-mapped data | Linux, read-only data, `fork` available |
| **`multiprocessing.shared_memory`** | Explicit shared buffer, both processes map it | Cross-platform, mutable shared state |
| **Pass filenames** | Each worker loads its own slice from disk | Data already on disk, or too big for RAM |

```python
# Pattern 1 — indices (Linux, fastest)
ex.map(partial_sum, [(s, e), ...])

# Pattern 2 — shared_memory (cross-platform)
from multiprocessing import shared_memory
shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
arr[:] = data
ex.map(partial_sum, [(shm.name, s, e), ...])

# Pattern 3 — filenames (robust)
ex.map(load_and_process, ["chunk_0.npy", "chunk_1.npy", ...])
```

> Same principle as Amdahl: **data movement is the serial fraction.**
> Minimise it — ideally to zero.

---

## The Compute Task (Not Memory-Bound!)

```python
def transform(x):
    """~15-20 FLOPs per element. Compute-heavy, NOT bandwidth-bound."""
    return np.sin(x) * np.cos(x) + x**2 - np.sqrt(np.abs(x) + 1e-9)
```

> Why not just `np.sum(data**2)`?
> Because `x**2` + `sum` on 20M floats is **memory-bandwidth-bound**.
> It saturates DRAM bandwidth on one core, so splitting across cores
> gives no speedup even without overhead.
>
> `transform()` does enough arithmetic per element that compute time
> dominates memory access time → parallelism actually helps.

<!-- note:
This is a subtle but important point for an HPC course.
If the task is memory-bound, adding cores doesn't help because
all cores fight for the same DRAM bandwidth.
The transform function has ~15-20 FLOPs per element, which means
the arithmetic intensity is high enough that compute dominates.
This is the same roofline-model reasoning from Week 1.
-->

---

## Live Demo: Scaling Workers (Correct Version)

> 🛠 **Run:** `python class-examples/week03/parallel_sum.py`

```text
Data size: 20,000,000 elements
Expected:  6841963.284712

Workers   Time (s)    Speedup     Efficiency
──────────────────────────────────────────────────
serial    1.842       1.00×       100%
2         0.964       1.91×       96%
4         0.512       3.60×       90%
8         0.298       6.18×       77%
```

> **Observations:**
> - Nearly linear up to 4 workers.
> - Diminishing returns at 8 (process creation + combine step).
> - This is Amdahl's Law in action.

<!-- note:
Run this live. 
- 1→2: nearly 2× (great!)
- 4→8: less than 2× (diminishing returns)
- The serial fraction includes: process creation, splitting bounds,
  sum(partials) combine step, and Python overhead.
"What would happen with 64 workers?" → overhead dominates.
-->

---

## Amdahl's Law: We Just Saw It

$$S(N) = \frac{1}{(1 - P) + \dfrac{P}{N}}$$

From our measurements:

| $N$ | Measured | Amdahl ($P = 0.97$) |
|:---:|:---:|:---:|
| 1 | 1.0× | 1.0× |
| 2 | 1.9× | 1.97× |
| 4 | 3.6× | 3.77× |
| 8 | 6.2× | 6.83× |
| $\infty$ | — | **33×** ceiling |

> Even with 97% parallel code, the 3% serial part caps you at ~33×.
> **You can never go faster than $\frac{1}{1-P}$.**

---

## Threads vs. Processes: The Full Picture

> 🛠 **Run:** `python parallel_sum.py --compare`

We compare **three** scenarios to see where GIL matters:

| Experiment | Threads help? | Processes help? | Why? |
|---|:---:|:---:|---|
| **Pure Python loop** (`for i: total += i*i`) | ❌ No | ✅ Yes | GIL blocks threads |
| **NumPy operation** (`np.sin(x)*np.cos(x)+...`) | ✅ Yes! | ✅ Yes | NumPy releases GIL |
| **I/O-bound** (`time.sleep`, file read) | ✅ Yes | ✅ Yes | GIL released during wait |

> **The GIL blocks PYTHON BYTECODE, not C-level operations.**
> NumPy's inner loops are C code that **release the GIL**,
> so threads CAN run NumPy operations in parallel.

<!-- note:
The simple rule "threads don't
help for CPU-bound" is WRONG. The correct rule:
- Threads don't help for PYTHON-LEVEL CPU-bound code (GIL held).
- Threads DO help for C-extension CPU-bound code (GIL released).
- NumPy, BLAS, PyTorch, scipy all release the GIL during compute.
- Pure Python loops, string processing, dict manipulation: GIL held.
This is why PyTorch DataLoader uses processes (augmentation has Python code)
but BLAS-based matmul can benefit from threads.
-->

---

## Experiment A: Pure Python Loop (GIL Blocks Threads)

```python
def cpu_work_python(n):
    """Pure Python. GIL held throughout."""
    total = 0
    for i in range(n):
        total += i * i
    return total
```

```text
Workers │ Processes  │ Threads    │ Note
────────┼────────────┼────────────┼─────────────────────
   1    │  2.410 s   │  2.430 s   │ Same (one worker)
   4    │  0.650 s   │  2.510 s   │ GIL blocks threads!
   8    │  0.380 s   │  2.580 s   │ Threads: NO speedup
```

> Threads: **no speedup** (GIL serialises Python bytecode).
> Processes: **~6× speedup** (each has its own GIL).

---

## Experiment B: NumPy Operations (GIL Released!)

```python
def cpu_work_numpy(bounds):
    """NumPy ops. GIL released during C-level computation."""
    start, end = bounds
    x = data[start:end]
    return float(np.sum(np.sin(x) * np.cos(x) + x**2))
```

```text
Workers │ Processes  │ Threads    │ Note
────────┼────────────┼────────────┼─────────────────────────────
   1    │  1.842 s   │  1.850 s   │ Same (one worker)
   4    │  0.512 s   │  0.540 s   │ Both give ~3.5× speedup!
   8    │  0.298 s   │  0.320 s   │ Threads work here!
```

> **Threads give speedup!** NumPy releases the GIL during
> `np.sin`, `np.cos`, `**`, `np.sum`. The C code runs truly in parallel.
> Processes also work (each has own GIL), but with more overhead.

<!-- note:
 threads don't help for PYTHON CPU-bound code.
But NumPy's inner loop is C code that releases the GIL.
So threads CAN parallelise NumPy operations.
The rule is more nuanced than "threads = I/O, processes = CPU."
The real rule: "threads work when the GIL is released."
In AI: PyTorch's CPU tensor operations release the GIL.
So CPU-side preprocessing with tensor ops CAN benefit from threads.
But Python-level control flow (if/else, string parsing) cannot.
-->

---

## Summary: When Do Threads Help?

```text
Is the operation releasing the GIL?
    │
    ├── YES (NumPy, BLAS, PyTorch C ops, I/O, sleep)
    │   └── Threads work! Use ThreadPoolExecutor.
    │
    └── NO (pure Python loops, string ops, dict manipulation)
        └── Threads DON'T help (GIL serialises).
            └── Use ProcessPoolExecutor instead.
```

| Operation | GIL held? | Threads parallel? |
|-----------|:---------:|:-----------------:|
| `for i in range(n): x += i` | ✅ Held | ❌ No |
| `np.sin(arr)` | ❌ Released | ✅ Yes |
| `arr @ arr.T` (BLAS) | ❌ Released | ✅ Yes |
| `time.sleep(1)` | ❌ Released | ✅ Yes |
| `open(f).read()` | ❌ Released | ✅ Yes |
| `json.loads(s)` | ✅ Held | ❌ No |

> **Memorise the principle:** GIL blocks *Python bytecode*.
> C extensions that release it enable true thread parallelism.

---

## 🚨 But Wait... Let's Check the Answer

> 🛠 **Run:** `python parallel_sum.py --check-correctness`

```text
Expected: 6841963.284712

  4 workers, Processes: 6841963.284712 ✓
  4 workers, Threads:   6841963.284712 ✓
  8 workers, Processes: 6841963.284712 ✓
  8 workers, Threads:   6841963.284712 ✓
```

> ✅ Both give the correct answer here.
> **Why?** Because each worker **returns** its result.
> No shared mutable state. No race condition.

**But what if we write to a shared global instead?**

---

## The "Optimised" Version: Shared Global Accumulator

> 🛠 **Run:** `class-examples/week03/parallel_sum_racy.py`

```python
import threading
import numpy as np

data = np.random.rand(10_000_000)
result = 0.0  # ← SHARED global accumulator

def partial_sum(start, end):
    global result
    local = np.sum(data[start:end] ** 2)
    result += local  # ← RACE CONDITION! Multiple threads write here.

threads = []
n_workers = 8
chunk = len(data) // n_workers
for i in range(n_workers):
    s, e = i * chunk, (i+1) * chunk if i < n_workers-1 else len(data)
    t = threading.Thread(target=partial_sum, args=(s, e))
    threads.append(t)
    t.start()
for t in threads:
    t.join()

expected = np.sum(data ** 2)
print(f"Expected: {expected:.6f}")
print(f"Got:      {result:.6f}")
print(f"Correct:  {np.isclose(result, expected)}")
```

---

## Live Demo: The Answer Is WRONG

> 🛠 **Run it 5 times:**

```text
$ python parallel_sum_racy.py
Expected: 3333328.472913
Got:      3333328.472913
Correct:  True

$ python parallel_sum_racy.py
Expected: 3333328.472913
Got:      2916654.104287
Correct:  False  ← !!!

$ python parallel_sum_racy.py
Expected: 3333328.472913
Got:      3124991.558342
Correct:  False  ← Different wrong answer!

$ python parallel_sum_racy.py
Expected: 3333328.472913
Got:      3333328.472913
Correct:  True   ← Sometimes "works" by luck
```

> **Different wrong answer every run.** Sometimes correct by accident.
> This is the hallmark of a **race condition**.

<!-- note:

1. "Wow, parallelism is fast!" (performance demo)
2. "But the answer is WRONG!" (this demo)

`result += local` is not atomic. It's LOAD, ADD, STORE.
Two threads can LOAD the same value, both ADD, both STORE → one lost.
With 8 threads, this happens frequently.
Sometimes all threads happen to execute sequentially (no interleaving) → correct.
Most of the time, they interleave → wrong.
-->

---

## What Just Happened?

```text
Thread 1: LOAD result → R1     (result = 100.0)
Thread 2: LOAD result → R2     (result = 100.0)  ← same value!
Thread 1: ADD R1 + 50 → R1     (R1 = 150.0)
Thread 2: ADD R2 + 30 → R2     (R2 = 130.0)
Thread 1: STORE R1 → result    (result = 150.0)
Thread 2: STORE R2 → result    (result = 130.0)  ← overwrites!

Should be: 100 + 50 + 30 = 180
Actually:  130  (Thread 1's +50 was LOST)
```

> `result += local` is **three operations**, not one.
> Between any two, another thread can sneak in.
> **This is a race condition.**

---

## The Question That Drives the Rest of Today

> We made it **fast** (parallelism works).
> We made it **wrong** (race condition).
>
> **How do we make it both fast AND correct?**

→ **Synchronisation.** (Part III)

---

## Summary of Part II

| What we showed | How | Result |
|---|---|---|
| Naive multiprocessing is SLOW | Pass 160 MB chunks via pickle | 8× slower than serial |
| Fix: pass indices, not data | `fork()` + bounds | ~6× speedup |
| Amdahl's Law is real | Speedup curve flattens | Diminishing returns |
| Pure Python: GIL blocks threads | `for` loop + threads | No speedup |
| NumPy: GIL released → threads work | `np.sin`, `np.sum` + threads | ~3.5× speedup |
| Shared state + parallelism = 💥 | Global accumulator, 8 threads | Wrong answers |

> **Next:** How to fix the correctness problem without losing the speed.


## Class-examples: `class-examples/week03/parallel_sum.py`

```python

```

---

## Class-examples: `class-examples/week03/parallel_sum_racy.py`

```python

```

<!-- _class: lead -->

# Part III
# Race Conditions: When Parallelism Breaks Correctness

---

## The Problem: Shared Mutable State

### C example (from your OS notes):

```c
#include <pthread.h>
int sum = 0;  // shared global

void *countgold(void *param) {
    for (int i = 0; i < 10000000; i++) {
        sum += 1;   // ← NOT atomic!
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, countgold, NULL);
    pthread_create(&t2, NULL, countgold, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("sum = %d\n", sum);  // Expected: 20000000
}
```

```text
$ ./a.out
sum = 10214142    ← WRONG!
$ ./a.out
sum = 11798888    ← Different wrong answer each run!
```

<!-- note:
(compile with gcc -pthread).
The answer is different every time. That's the hallmark of a race condition.
 "Why isn't it 20 million?"
Answer: sum += 1 is not one instruction. It's three steps.
-->

---

## Why: `sum += 1` Is Three Steps

```text
Machine level:
  Step 1: LOAD   sum → register     (read from memory)
  Step 2: ADD    register + 1       (compute)
  Step 3: STORE  register → sum     (write back)
```

**Interleaving that loses an update:**

```text
Time   Thread 1              Thread 2              sum in memory
────   ────────              ────────              ─────────────
 t0    LOAD sum → R1                               0
 t1    (R1 = 0)
 t2                          LOAD sum → R2         0
 t3                          (R2 = 0)
 t4    ADD R1+1 → R1
 t5    (R1 = 1)
 t6                          ADD R2+1 → R2
 t7                          (R2 = 1)
 t8    STORE R1 → sum                              1
 t9                          STORE R2 → sum        1  ← SHOULD BE 2!
```

> Both threads read 0, both computed 1, both wrote 1. **One update lost.**
> With 10 million iterations, thousands of updates are lost.

---

## Python Equivalent (Same Bug)

```python
import threading

counter = 0  # shared global

def increment():
    global counter
    for _ in range(10_000_000):
        counter += 1  # NOT atomic in Python either!

t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)
t1.start(); t2.start()
t1.join();  t2.join()

print(f"Expected: 20,000,000")
print(f"Actual:   {counter:,}")  # ~12,000,000 – 18,000,000
```

> Same problem. `counter += 1` compiles to LOAD, ADD, STORE bytecode.
> The GIL doesn't fully protect us here. (More in Part V.)

---

## Race Condition on a NumPy Array (AI Context)

```python
import numpy as np
import threading

# Shared global array (e.g., gradient accumulator)
grad_accumulator = np.zeros(1000)

def accumulate_gradients(grads):
    global grad_accumulator
    for i in range(len(grads)):
        grad_accumulator[i] += grads[i]  # ← race condition!

# Two threads accumulating different gradients
g1 = np.random.rand(1000)
g2 = np.random.rand(1000)

t1 = threading.Thread(target=accumulate_gradients, args=(g1,))
t2 = threading.Thread(target=accumulate_gradients, args=(g2,))
t1.start(); t2.start()
t1.join();  t2.join()

expected = g1 + g2
print(f"Max error: {np.max(np.abs(grad_accumulator - expected)):.6f}")
# Often non-zero! Some elements have lost updates.
```

> **In AI:** this is what happens if two backward passes write to `.grad`
> simultaneously without synchronisation. PyTorch prevents this internally,
> but custom training loops can hit it.

<!-- note:
This connects directly to AI. Gradient accumulation across multiple
backward passes, or multiple workers updating a shared statistics
array (e.g., running mean in batch norm), can race.
The fix: lock, or use atomic operations, or restructure to avoid sharing.
-->

---

## Race Condition: Formal Definition

> A **race condition** occurs when:
> 1. Two or more threads access **shared state**, AND
> 2. At least one access is a **write**, AND
> 3. The accesses are **not synchronised**.

If any condition is absent → no race.

| Scenario | Race? |
|----------|:-----:|
| Two threads **read** the same array | ❌ No write |
| Two threads write to **different** arrays | ❌ Not shared |
| Two threads write to the **same** array, no lock | ✅ Race! |
| Two threads write to same array, **with lock** | ❌ Synchronised |

---

## The Critical Section

> The **critical section** is the code region that accesses shared state.

```python
# ─── BEGIN critical section ───
grad_accumulator[i] += grads[i]
# ─── END critical section ───
```

**Goal of synchronisation:** ensure at most **one thread** is inside
the critical section at any time (**mutual exclusion**).

```text
Thread 1:  [  CS  ]         [  CS  ]
Thread 2:          [  CS  ]         [  CS  ]
           ←─ mutual exclusion ─→
           (never overlapping)
```

> The rest of today: **how** to enforce mutual exclusion.

---

<!-- _class: lead -->

# Part IV
# Synchronisation Primitives

---

## Mutex (Lock): The Basic Tool

> A **mutex** (mutual exclusion lock) allows only **one thread**
> to hold the lock at a time. Others **block** until it's released.

### C (POSIX):

```c
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

void *countgold(void *param) {
    for (int i = 0; i < 10000000; i++) {
        pthread_mutex_lock(&m);     // acquire
        sum += 1;                   // critical section
        pthread_mutex_unlock(&m);   // release
    }
    return NULL;
}
```

### Python equivalent:

```python
import threading
lock = threading.Lock()
counter = 0

def increment():
    global counter
    for _ in range(10_000_000):
        with lock:              # acquire + auto-release
            counter += 1
```

> Correct result: 20,000,000. But **much slower** (serialised).

---

## Mutex: The "Mic" Analogy

> Think of a mutex as a **microphone** in a discussion.
> - Whoever holds the mic gets to speak (access shared data).
> - Others must **wait** until the mic is free.
> - You must **put the mic back** when done (unlock).
> - If you forget to put it back → everyone waits forever (**deadlock**).

```text
Thread 1: 🎤 "I'm updating sum..."     [puts mic back]
Thread 2:    (waiting...)              🎤 "Now I update sum..."
Thread 3:    (waiting...)              (waiting...)
```

---

## Performance: Lock Granularity Matters

### Bad: lock/unlock every iteration (10M lock operations!)

```c
for (int i = 0; i < 10000000; i++) {
    pthread_mutex_lock(&m);
    sum += 1;
    pthread_mutex_unlock(&m);
}
```

### Good: accumulate locally, lock once at the end

```c
void *countgold(void *param) {
    int local_sum = 0;
    for (int i = 0; i < 10000000; i++) {
        local_sum += 1;          // no lock needed (local variable)
    }
    pthread_mutex_lock(&m);
    sum += local_sum;            // one lock operation
    pthread_mutex_unlock(&m);
    return NULL;
}
```

> **10,000,000× fewer lock operations.** Massive speedup.
> **Principle:** do work outside the lock; lock only for the shared update.

<!-- note:
The key insight:
- Local variables are on the thread's private stack → no race.
- Only the final accumulation into the global needs the lock.
- This reduces critical section time from O(n) to O(1).
In AI: compute gradients locally (on each GPU), then synchronise once.
That's exactly what DDP does (Week 10).
-->

---

## NumPy Example: Parallel Partial Sums

```python
import numpy as np
import threading

data = np.random.rand(10_000_000)
result = 0.0
lock = threading.Lock()

def partial_sum(start, end):
    global result
    local = np.sum(data[start:end])   # compute locally (no lock)
    with lock:
        result += local               # one lock operation

threads = []
n_chunks = 4
chunk_size = len(data) // n_chunks
for i in range(n_chunks):
    s, e = i * chunk_size, (i + 1) * chunk_size
    t = threading.Thread(target=partial_sum, args=(s, e))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Parallel sum: {result:.6f}")
print(f"NumPy sum:    {np.sum(data):.6f}")
```

> Compute locally → lock once to combine. Same pattern as the C example.

---

## Semaphore: Allow N Threads Simultaneously

> A **semaphore** generalises a mutex: allows up to **N** concurrent accesses.

### C (POSIX):

```c
sem_t s;
sem_init(&s, 0, 3);  // 3 "permits" available

void *worker(void *arg) {
    sem_wait(&s);     // decrement; blocks if 0
    // ... use resource (at most 3 threads here) ...
    sem_post(&s);     // increment; may wake a waiting thread
    return NULL;
}
```

### Python:

```python
import threading
sem = threading.Semaphore(3)  # at most 3 concurrent

def worker(id):
    with sem:
        print(f"Thread {id} using resource")
        # ... at most 3 threads in here at once ...
```

> **AI use case:** "I have 2 GPUs. At most 2 training jobs at once."

---

## Semaphore vs. Mutex

| | Mutex (Lock) | Semaphore |
|---|---|---|
| Allows | **1** thread in CS | **N** threads simultaneously |
| Ownership | Locking thread must unlock | Any thread can `sem_post` |
| Use case | Protect one shared variable | Limit concurrent access to N resources |
| Python | `threading.Lock()` | `threading.Semaphore(N)` |

> A mutex is a semaphore with N=1 **plus** ownership tracking.

---

## Condition Variable: Wait for a Signal

> A **condition variable** lets a thread **sleep** until another thread
> signals that a condition is now true.

### Pattern:

```python
import threading

condition = threading.Condition()
queue = []

def producer():
    with condition:
        queue.append(new_item)
        condition.notify()       # "Hey consumer, there's data!"

def consumer():
    with condition:
        while not queue:         # use while, not if (spurious wakeups)
            condition.wait()     # sleep until notified
        item = queue.pop(0)
        # process item
```

### C (POSIX):

```c
pthread_mutex_lock(&m);
while (count == 0)
    pthread_cond_wait(&cv, &m);  // sleep, release mutex
// ... consume ...
pthread_mutex_unlock(&m);
```

> **Key:** `wait()` atomically releases the mutex and sleeps.
> When woken, it re-acquires the mutex before returning.

---

## Producer–Consumer: The Full Pattern

```text
┌──────────┐     ┌──────────────────────────────┐     ┌──────────┐
│ Producer │ ──→ │  BOUNDED BUFFER (size N)     │ ──→ │ Consumer │
│ (thread) │put()│  [item][item][item][  ][  ]  │get()│ (thread) │
└──────────┘     └──────────────────────────────┘     └──────────┘
                  blocks if FULL          blocks if EMPTY
```

```python
import queue
q = queue.Queue(maxsize=10)  # thread-safe, handles all locking

# Producer
q.put(item)       # blocks if queue full

# Consumer
item = q.get()    # blocks if queue empty
```

> `queue.Queue` uses a mutex + two condition variables internally.
> You just call `put()` and `get()`. All synchronisation is hidden.

> **In AI:** `DataLoader(num_workers=4)` uses exactly this pattern.
> Workers produce batches → queue → training loop consumes.

---

## Barrier: All Threads Wait Here

> A **barrier** blocks all threads until every thread has arrived.

### C (POSIX):

```c
pthread_barrier_t barrier;
pthread_barrier_init(&barrier, NULL, 4);  // 4 threads

void *worker(void *arg) {
    do_phase_1();
    pthread_barrier_wait(&barrier);  // block until all 4 arrive
    do_phase_2();                    // all start together
}
```

### Python:

```python
barrier = threading.Barrier(4)

def worker(id):
    do_phase_1(id)
    barrier.wait()       # all 4 must reach here
    do_phase_2(id)       # then all proceed together
```

> **AI use case:** "All workers must finish loading before any starts computing."
> Distributed training: all GPUs must finish backward before optimiser step.

---

## Synchronisation Primitives: Summary

| Primitive | Allows | Blocks when | Python |
|-----------|--------|-------------|--------|
| **Lock / Mutex** | 1 thread in CS | Lock already held | `threading.Lock()` |
| **Semaphore(N)** | N threads | 0 permits left | `threading.Semaphore(N)` |
| **Condition** | Wait for signal | Condition false | `threading.Condition()` |
| **Barrier(N)** | Sync N threads | Not all arrived | `threading.Barrier(N)` |
| **Queue** | Producer-consumer | Full / empty | `queue.Queue(maxsize)` |

> In AI frameworks, these are used **internally**. You'll rarely write
> raw locks. But you need the concepts to understand what's happening
> and to debug when things go wrong.

---

## Deadlock: When Synchronisation Goes Wrong

> **Deadlock:** two or more threads wait for each other **forever**.

```text
Thread 1: holds Lock A, wants Lock B
Thread 2: holds Lock B, wants Lock A

Thread 1: "I'll release A after I get B"
Thread 2: "I'll release B after I get A"
→ Neither proceeds. 💀
```

### C:

```c
// Thread 1                    // Thread 2
pthread_mutex_lock(&A);       pthread_mutex_lock(&B);
pthread_mutex_lock(&B); 💀    pthread_mutex_lock(&A); 💀
```

### Prevention rules:
1. **Always acquire locks in the same order** across all threads.
2. Use `with` statements (auto-release, even on exception).
3. Avoid holding multiple locks simultaneously.
4. Use timeouts: `lock.acquire(timeout=5.0)`.

---

## Dining Philosophers (Classic Deadlock)

```text
        P0
    C4 /  \ C0
      P4    P1
      |      |
    C3 \  / C1
        P3──P2
          C2

5 philosophers, 5 chopsticks. Each needs 2 to eat.
All grab left chopstick simultaneously → DEADLOCK.
```

### Solution (Dijkstra): **hierarchical ordering**

```c
// Always pick up the LOWER-numbered chopstick first
int first  = MIN(i, (i+1) % 5);
int second = MAX(i, (i+1) % 5);
pthread_mutex_lock(&chopsticks[first]);
pthread_mutex_lock(&chopsticks[second]);
// eat
pthread_mutex_unlock(&chopsticks[second]);
pthread_mutex_unlock(&chopsticks[first]);
```

> Breaks the **circular wait** condition. Deadlock impossible.

<!-- note:
The key takeaway:
"Always acquire resources in a consistent global order."
This applies to GPU locks, file handles, database connections.
 Just understand the principle.
-->

---

## Design Principles for Synchronisation

1. **Minimise critical sections.** Do computation outside the lock.
2. **One lock per shared resource.** Don't use one global lock for everything.
3. **Lock ordering.** If you need multiple locks, always acquire in the same order.
4. **Prefer high-level constructs.** `queue.Queue` > manual lock + condition.
5. **Avoid shared mutable state** when possible.
   - Pass data by value (no sharing).
   - Use return values instead of global variables.
   - Use thread-local storage.

> In AI: PyTorch handles most synchronisation internally.
> Your job: understand the concepts so you can reason about
> DataLoader, DDP, and custom pipelines.

---

<!-- _class: lead -->

# Part V
# Python's Twist: The GIL as Accidental Protector

---

## The GIL: Quick Preview (Deep Dive Next Week)

> CPython's **Global Interpreter Lock** allows only one thread
> to execute Python bytecode at a time.

This has a surprising side effect:

**Some Python operations are accidentally thread-safe** because the GIL
makes them effectively atomic (they complete in one bytecode instruction).

```python
# These are single bytecode operations → GIL makes them atomic:
my_list.append(x)       # atomic (one bytecode: LIST_APPEND)
my_dict[key] = value    # atomic (one bytecode: STORE_SUBSCR)
x = my_list.pop()       # atomic (one bytecode: LIST_POP)
```

> In C, you'd need a mutex for all of these.
> In CPython, the GIL accidentally protects them.

---

## But: Compound Operations Are NOT Protected

```python
counter += 1   # NOT atomic! (LOAD, ADD, STORE = 3 bytecodes)
```

```text
Bytecode for `counter += 1`:
  LOAD_GLOBAL   counter     ← GIL could switch here!
  LOAD_CONST    1
  BINARY_ADD                ← or here!
  STORE_GLOBAL  counter
```

Between any two bytecodes, the GIL can switch to another thread.
So `counter += 1` is **NOT safe** even with the GIL.

> **The GIL is NOT a substitute for proper synchronisation.**
> It accidentally protects simple operations, but not compound ones.

---

## Demonstration: GIL Helps vs. Doesn't Help

```python
import threading

# SAFE (single bytecode, GIL-protected):
shared_list = []
def append_items():
    for i in range(100000):
        shared_list.append(i)  # atomic → no race

# UNSAFE (multiple bytecodes, GIL doesn't help):
counter = 0
def increment():
    global counter
    for _ in range(100000):
        counter += 1  # NOT atomic → race condition!
```

| Operation | Atomic under GIL? | Needs explicit lock? |
|-----------|:-----------------:|:--------------------:|
| `list.append(x)` | ✅ Yes | ❌ No |
| `dict[k] = v` | ✅ Yes | ❌ No |
| `counter += 1` | ❌ No | ✅ Yes |
| `arr[i] += val` (NumPy) | ❌ No (GIL released!) | ✅ Yes |

---

## NumPy Releases the GIL → Race Conditions Return!

```python
import numpy as np
import threading

arr = np.zeros(1_000_000)

def add_to_array():
    for i in range(len(arr)):
        arr[i] += 1.0  # NumPy releases GIL during this!

t1 = threading.Thread(target=add_to_array)
t2 = threading.Thread(target=add_to_array)
t1.start(); t2.start()
t1.join();  t2.join()

print(f"Expected: all elements = 2.0")
print(f"Actual:   min={arr.min()}, max={arr.max()}")
# min < 2.0! Some increments were lost.
```

> When NumPy operates on arrays, it **releases the GIL**
> (so other threads can run). This means:
> - Good for parallelism (threads can truly run concurrently).
> - Bad for safety (race conditions on shared arrays are possible).

> **In AI:** if two threads write to the same tensor simultaneously,
> you get corruption. PyTorch prevents this internally, but custom
> code must be careful.

---

## Summary: GIL Protection Map

```text
┌─────────────────────────────────────────────────────────────────┐
│  Operation                    │ GIL held? │ Thread-safe?        │
├─────────────────────────────────────────────────────────────────┤
│  list.append(x)               │ ✅ Yes    │ ✅ Accidentally safe │
│  dict[k] = v                  │ ✅ Yes    │ ✅ Accidentally safe │
│  counter += 1                 │ ✅ Yes    │ ❌ NOT safe (3 steps)│
│  numpy arr[i] += val          │ ❌ Released│ ❌ NOT safe         │
│  torch tensor ops             │ ❌ Released│ ❌ NOT safe         │
│  time.sleep() / file I/O      │ ❌ Released│ ✅ (no shared state)│
└─────────────────────────────────────────────────────────────────┘
```

> **The GIL is NOT a synchronisation mechanism.**
> It's a memory-management implementation detail that accidentally
> makes *some* operations atomic. Never rely on it for correctness.

> **Next week:** GIL in depth, `concurrent.futures`, when threads
> vs. processes matter, and how to parallelise correctly in Python.

---

## Key Takeaways – Week 3

1. **Why parallelism:** exploit multiple cores for data/task parallelism.
   Matrix ops, batch processing, pipeline stages — all parallelisable.

2. **Threads share memory.** That's powerful (fast communication)
   but dangerous (race conditions).

---

3. **Race conditions** happen when unsynchronised threads access
   shared mutable state. The fix: **mutual exclusion** (locks).

4. **Synchronisation primitives:** lock, semaphore, condition, barrier, queue.
   Keep critical sections small. Accumulate locally, lock once.

---

5. **Python's GIL** accidentally protects some operations but is NOT
   a substitute for proper synchronisation. NumPy/C extensions release
   the GIL → race conditions are real.

6. **In AI:** DataLoader uses processes + queue. DDP uses locks + barriers.
   Understanding these concepts lets you reason about the full stack.

---

## Before Next Week

**Week 4:** Python Parallelism in Practice
- The GIL in depth: what it blocks, what it doesn't
- `concurrent.futures`: `ThreadPoolExecutor` / `ProcessPoolExecutor`
- I/O-bound vs. CPU-bound: choosing the right tool
- Async / coroutines (brief intro)
- Connection to `DataLoader`, Ray, distributed training
- **Hands-on lab**

**Optional reading (15 min):**
- Python `threading` docs: https://docs.python.org/3/library/threading.html
- Python `queue` docs: https://docs.python.org/3/library/queue.html

**Think about:**
- Where in your thesis/project code do you have independent tasks
  that could run in parallel?
- Is the bottleneck I/O (loading data) or CPU (computation)?

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank You

### Next week: The GIL, `concurrent.futures`, and Python parallelism in AI practice

*YZM 513 – İstanbul Medeniyet Üniversitesi – Fall 2026*


## Class-examples: `class-examples/week03/`

### `race_condition_counter.py`

```python

```

### `race_condition_numpy.py`

```python

```

### `partial_sum_parallel.py`

```python

```

### `producer_consumer_queue.py`

```python

```

### `semaphore_gpu_limit.py`

```python

```

### `barrier_sync.py`

```python

```

---


