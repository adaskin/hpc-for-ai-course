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
*prepared with Qwen based on my prompts and notes*

### : Week 4: Python Parallelism in Practice
### The GIL, `concurrent.futures`, Async I/O & AI Workloads

**İstanbul Medeniyet Üniversitesi**
AI Engineering M.Sc. – Fall 2026

Ammar Daşkın

---

## Recap: Where We Are

**Week 3 concepts (you know these now):**
- Data parallelism vs. task parallelism
- Threads share memory → race conditions
- Locks, semaphores, condition variables, barriers, queues
- C examples → Python equivalents
- GIL as "accidental protector" (preview)

**Today:**
- The GIL **in depth**: what it blocks, what it doesn't, why it matters for AI
- `concurrent.futures`: the practical API you'll actually use
- Async / coroutines for massive I/O (LLM serving, agent swarms)
- Thread/process pools as AI infrastructure
- **All examples are AI workloads.** No generic "sum of squares."

---

## Today's Arc

```text
Part I:   The GIL in depth
          → WHY Python threading is weird
          → What blocks, what doesn't

Part II:  concurrent.futures
          → ThreadPoolExecutor for I/O (LLM API calls)
          → ProcessPoolExecutor for CPU (tokenization, augmentation)
          → submit + as_completed (hyperparameter sweep)

Part III: Async / coroutines
          → LLM batch inference, agent swarms, streaming

Part IV:  Pools as AI infrastructure
          → DataLoader, inference servers

Part V:   Lab
```

---

<!-- _class: lead -->

# Part I: The GIL in Depth

---

## The GIL: One Sentence

> **The Global Interpreter Lock is a single mutex inside CPython
> that allows only ONE thread to execute Python bytecode at a time.**

---

```text
┌─────────────────────────────────────────────────────────┐
│  CPython Process (e.g., your training script)           │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │  THE GIL (one lock for the entire process)        │ │
│  │                                                   │ │
│  │  Thread 0 (data loader):  ████░░░░████░░░░████   │ │
│  │  Thread 1 (augmenter):    ░░░░████░░░░████░░░░   │ │
│  │  Thread 2 (logger):       ░░░░░░░░░░░░░░░░████   │ │
│  │                                                   │ │
│  │  ████ = holding GIL (executing Python bytecode)   │ │
│  │  ░░░░ = waiting for GIL (blocked)                 │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Even on an 8-core CPU: ONE thread runs Python at a time│
└─────────────────────────────────────────────────────────┘
```

<!-- note:
"You learned threads run in parallel on multiple cores.
In CPython, that's true for C code, I/O, CUDA calls. But for PYTHON BYTECODE,
only one thread at a time. That's the GIL."
 "Why does Python do this? Why not just let threads run free?"
-->

---

## Why Does the GIL Exist?

CPython manages memory with **reference counting**:

```python
model = load_model()     # refcount(model) = 1
trainer = Trainer(model) # refcount(model) = 2
del trainer              # refcount(model) = 1
del model                # refcount(model) = 0 → free memory
```

- Every Python object tracks how many references point to it.
- Without the GIL: two threads modify refcounts simultaneously → **memory corruption**.
- The GIL protects ALL interpreter state with **one big lock**.

---

**Design trade-off:**

| Pro | Con |
|-----|-----|
| Simple implementation | Multi-threaded CPU-bound Python is serialised |
| Single-threaded code has zero lock overhead | Can't exploit multiple cores for Python loops |
| C extensions can opt in/out | Accidental complexity for concurrent code |

> The GIL is a **CPython implementation detail**, not a Python language feature.

<!-- note:
Key point: the GIL exists to make CPython's memory management simple.
Per-object locks would be correct but add overhead to EVERY operation,
even single-threaded code. The GIL trades multi-thread performance for
single-thread simplicity.
This is why NumPy, PyTorch, etc. release the GIL: they manage their own
memory (C/C++ side) and don't need CPython's refcounting protection.
-->

---

## GIL Mechanics: The Switch Interval

The GIL is not held forever. CPython **forces a thread to release it** periodically:

```python
import sys
print(sys.getswitchinterval())  # default: 0.005 (5 ms)
```

```text
Timeline (5ms switch interval):

Thread A: █████░░░░░░░░░░█████░░░░░░░░░░█████░░░░░
Thread B: ░░░░░█████░░░░░░░░░░░█████░░░░░░░░░░█████
Thread C: ░░░░░░░░░░░█████░░░░░░░░░░░█████░░░░░░░░░
          |←5ms→|
          GIL switches every ~5ms (or on I/O, or on explicit release)
```

- A thread holding the GIL for > 5ms is **asked** to release it.
- Another thread acquires it and runs.
- This gives the **illusion** of parallelism. But only one thread runs Python at a time.

> You can change it: `sys.setswitchinterval(0.001)` (1ms).
> Shorter interval → more context switches → more overhead → less throughput.

<!-- note:
change it and re-run a benchmark to see the effect.
The key insight: the GIL doesn't block forever; it time-slices. But time-slicing
is NOT parallelism. It's concurrency on one core.
-->

---

## What the GIL Blocks vs. What It Doesn't

| Operation | GIL held? | Threads parallel? | AI example |
|-----------|:---------:|:-----------------:|------------|
| Python arithmetic, loops | ✅ Yes | ❌ No | Custom training loop in pure Python |
| `time.sleep(n)` | ❌ Released | ✅ Yes | Simulated API latency |
| File I/O (`open`, `read`) | ❌ Released | ✅ Yes | Loading images from disk |
| Network I/O (`requests`, `aiohttp`) | ❌ Released | ✅ Yes | Calling LLM inference API |
| NumPy / BLAS operations | ❌ Released | ✅ Yes | Matrix multiply in a layer |
| PyTorch CUDA operations | ❌ Released | ✅ Yes | `model.forward()` on GPU |
| `torch.load()` / `torch.save()` | ❌ Released | ✅ Yes | Checkpoint I/O |
| C extensions that release GIL | ❌ Released | ✅ Yes | Tokenizers (HuggingFace `tokenizers` lib) |

> **The GIL blocks PYTHON BYTECODE. It does NOT block I/O or C/CUDA calls.**

<!-- note:
This is the most important table of the week. 
Key insight: "The GIL doesn't prevent ALL parallelism. It prevents
parallel PYTHON BYTECODE. When Python calls into C, CUDA, or does I/O,
it releases the GIL."
This is why:
- 4 threads calling an LLM API → 4× speedup (network I/O, GIL released)
- 4 threads running a Python tokenizer loop → NO speedup (GIL held)
- 4 threads calling np.dot() → speedup (NumPy releases GIL)
- 4 threads calling model.forward() on GPU → speedup (CUDA releases GIL)
-->

---

## Visual: I/O-Bound AI Task (GIL Released)

### Scenario: 4 threads querying an LLM inference server

```text
Serial (1 thread):
  [query prompt 1].....[query prompt 2].....[query prompt 3].....[query prompt 4].....
  |←── 200ms ──→|      |←── 200ms ──→|      |←── 200ms ──→|      |←── 200ms ──→|
  Total: 800ms

Parallel (4 threads):
  Thread 1: [query prompt 1].....
  Thread 2: [query prompt 2].....
  Thread 3: [query prompt 3].....
  Thread 4: [query prompt 4].....
            |←── 200ms ──→|
            Total: ~200ms (4× speedup!)
```

- During the network wait, the thread **releases the GIL**.
- Other threads acquire the GIL and issue their requests.
- All four HTTP calls happen **simultaneously**.

> **This is exactly what happens when you batch-query a vLLM / TGI server.**

<!-- note:
The dots represent waiting for the network. The GIL is NOT held during
the wait. All four threads are "blocked on I/O" simultaneously.
The CPU is idle during the wait → no GIL contention.
This is why ThreadPoolExecutor works great for API calls.
-->

---

## Visual: CPU-Bound AI Task (GIL Blocks)

### Scenario: 4 threads tokenizing documents with a pure-Python tokenizer

```text
Serial (1 thread):
  [tokenize doc 1]████████████████████████████
  |←──────────── 400ms ──────────────────→|

"Parallel" (4 threads, pure Python tokenizer):
  Thread 1: ████░░░░░░░░████░░░░░░░░████░░░░
  Thread 2: ░░░░████░░░░░░░░████░░░░░░░░████
  Thread 3: ░░░░░░░░████░░░░░░░░████░░░░░░░░
  Thread 4: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
            |←──────── ~1600ms ──────────→|
            WORSE than serial! (context-switch overhead)
```

- Only one thread executes Python bytecode at a time.
- Threads **take turns** every ~5ms.
- Extra threads add context-switch overhead.
- **Result: slower than single-threaded!**

> **Solution:** use `ProcessPoolExecutor` (each process has its own GIL)
> or use a C-based tokenizer (HuggingFace `tokenizers` releases the GIL).

<!-- note:
threads make CPU-bound Python SLOWER, not faster.
The fix:
1. ProcessPoolExecutor: each process has its own interpreter + GIL.
2. Use a C-based tokenizer (HuggingFace tokenizers library) which
   releases the GIL → threads work.
3. Use Numba @njit(parallel=True) from Week 2.
-->

---

## The Rule (Memorise This)

| AI Workload | Use | Why |
|-------------|-----|-----|
| Calling LLM inference API (network) | **Threads** | GIL released during I/O |
| Loading images / data shards from disk | **Threads** or **Processes** | GIL released during I/O |
| Tokenizing with pure-Python tokenizer | **Processes** | GIL blocks Python bytecode |
| Tokenizing with HF `tokenizers` (C/Rust) | **Threads** work | C code releases GIL |
| Image augmentation (NumPy/PIL) | **Threads** work | NumPy releases GIL |
| Running `model.forward()` on GPU | **Threads** work | CUDA releases GIL |
| Custom Python training loop (CPU) | **Processes** | GIL blocks Python bytecode |

> This table answers: "Should I use threads or processes for my AI task?"

---

## Live Demo: GIL in Action

> 📁 `class-examples/week04/01_gil_llm_vs_tokenize.py`

```python
# Two AI tasks, both with 4 threads:

# Task A: Query LLM server (I/O-bound) → threads help
def query_llm(prompt_id):
    time.sleep(0.2)  # simulated network + inference
    return f"response_{prompt_id}"

# Task B: Tokenize in pure Python (CPU-bound) → threads DON'T help
def tokenize_pure_python(text):
    tokens = []
    for char in text:
        if char == ' ':
            tokens.append('<SEP>')
        else:
            tokens.append(char)
    return tokens
```

**Expected results:**

| Task | Serial (1 thread) | 4 Threads | Speedup |
|------|------------------:|----------:|--------:|
| LLM query (I/O) | 0.80 s | 0.21 s | **~4×** ✓ |
| Tokenize (CPU) | 1.20 s | 1.35 s | **~0.9×** ✗ |

> Threads help for I/O. Threads **hurt** for CPU-bound Python.

<!-- note:
 "Why did the tokenizer get SLOWER with threads?"
Answer: GIL serialises + context-switch overhead.
 "How would you fix the tokenizer?"
Answer: ProcessPoolExecutor, or use HuggingFace tokenizers (C/Rust, releases GIL).
-->

---

## The GIL and PyTorch: What You Need to Know

```python
import torch

# This releases the GIL (CUDA operation):
output = model(input_tensor.cuda())

# This releases the GIL (BLAS on CPU):
output = torch.matmul(a, b)  # calls into MKL/OpenBLAS

# This HOLDS the GIL (Python control flow):
for epoch in range(100):
    for batch in dataloader:
        loss = model(batch)     # ← CUDA call releases GIL
        loss.backward()         # ← CUDA call releases GIL
        optimizer.step()        # ← CUDA call releases GIL
        # But the Python loop overhead (iteration, function calls)
        # still holds the GIL briefly between CUDA launches.
```

> **In practice:** PyTorch training loops are GPU-bound.
> The GIL is rarely the bottleneck. But data loading (CPU, Python) IS.
> That's why `DataLoader(num_workers=N)` uses **processes**, not threads.

<!-- note:
This connects to Week 9 (DataLoader optimisation).
The key insight: during model.forward() and backward(), the GIL is
released because CUDA operations happen in C++. But the Python-level
loop (iterating over batches, calling functions) briefly holds the GIL.
This is rarely the bottleneck when the GPU is busy. But when the GPU
is WAITING for data (because data loading is slow), the GIL + CPU
become the bottleneck. Hence: multi-process data loading.
-->

---

## Free-Threading: The Future (Brief)

Python 3.13+ has an **experimental** GIL-free build:

```bash
python3.13t script.py   # 't' = free-threaded
```

| | With GIL (standard) | Without GIL (3.13t) |
|---|:---:|:---:|
| CPU-bound threads | ❌ Serialised | ✅ True parallelism |
| Race conditions | Hidden by GIL | **Exposed!** |
| Need for locks (Week 3!) | Rarely needed | **Required** |

> **Removing the GIL doesn't make unsafe code safe.**
> It removes the accidental serialisation that hid your bugs.
> Everything from Week 3 (locks, race conditions) becomes **more important**.

> Status in 2026: stabilising, but not the default. NumPy/PyTorch support growing.

<!-- note:
 The key message: "The GIL's days are numbered.
But the concepts from Week 3 (locks, critical sections, race conditions)
become MORE important in a free-threaded world, not less.
If you write thread-safe code now, it works in both worlds."
-->

---

<!-- _class: lead -->

# Part II
# `concurrent.futures`: The Practical API

---

## Why Not Raw `threading.Thread`?

```python
# Raw threading: verbose, error-prone
import threading

results = [None] * 20
errors = [None] * 20

def query_llm(idx, prompt):
    try:
        results[idx] = call_api(prompt)
    except Exception as e:
        errors[idx] = e

threads = [threading.Thread(target=query_llm, args=(i, p))
           for i, p in enumerate(prompts)]
for t in threads: t.start()
for t in threads: t.join()
# Now check results and errors manually...
```

**Problems:**
- Manual thread management (start, join)
- Shared mutable list for results (race condition risk!)
- No built-in error propagation
- No way to limit concurrency (20 threads all start at once)
- No clean "get me results as they finish" pattern

> **`concurrent.futures`** solves all of this in 3 lines.

---

## The Executor Pattern

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Choose your executor based on the workload:
#   ThreadPoolExecutor  → I/O-bound (LLM API calls, file loading)
#   ProcessPoolExecutor → CPU-bound (tokenization, augmentation in pure Python)

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(query_llm, prompts))
```

- `executor.map(fn, iterable)` → applies `fn` to each item **in parallel**.
- Returns results **in order** (same as input).
- `with` block handles cleanup (join threads / terminate processes).
- `max_workers` limits concurrency.

> That's it. Three lines. No manual thread management. No shared lists.

---

## `executor.map()`: Batch LLM Inference

### Scenario: Send 20 prompts to an inference server, collect responses.

```python
from concurrent.futures import ThreadPoolExecutor
import time

def query_inference_server(prompt: str) -> str:
    """Simulate HTTP POST to vLLM/TGI at localhost:8000/generate."""
    time.sleep(0.2)  # simulated network + generation latency
    return f"Response to: '{prompt[:30]}...'"

prompts = [
    "Explain gradient descent in one sentence.",
    "What is the attention mechanism?",
    "Write a haiku about backpropagation.",
    "Compare Adam and SGD optimizers.",
    # ... 20 prompts total
]

# Serial: 20 × 0.2s = 4.0 s
# Parallel (10 threads): ~0.4 s

with ThreadPoolExecutor(max_workers=10) as ex:
    responses = list(ex.map(query_inference_server, prompts))

print(f"Got {len(responses)} responses")
```

> 📁 Full script: `class-examples/week04/02_batch_llm_inference.py`

<!-- note:
Key  points:
1. time.sleep simulates network I/O. GIL is released during sleep.
2. 10 workers means at most 10 concurrent requests.
3. Results come back IN ORDER (same order as prompts).
4. The `with` block ensures all threads are joined before continuing.
5. This is exactly what you'd do with a real vLLM server, just
   replace time.sleep with requests.post() or aiohttp.
-->

---

## `executor.submit()` + `as_completed()`: Hyperparameter Sweep

### Scenario: Run 8 training experiments with different configs.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import time, random

def train_experiment(config: dict) -> dict:
    """Simulate a training run. Returns val_loss."""
    time.sleep(2)  # pretend this is 2 hours of training
    val_loss = random.uniform(0.1, 0.9)
    return {"config": config, "val_loss": val_loss}

configs = [
    {"lr": lr, "batch_size": bs, "layers": L}
    for lr in [1e-3, 1e-4]
    for bs in [32, 64]
    for L in [4, 8]
]  # 8 configs

with ProcessPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(train_experiment, c): c for c in configs}
    for future in as_completed(futures):
        result = future.result()
        print(f"  lr={result['config']['lr']}, bs={result['config']['batch_size']}, "
              f"layers={result['config']['layers']} → loss={result['val_loss']:.3f}")
```

> 📁 Full script: `class-examples/week04/03_hyperparam_sweep.py`

<!-- note:
Key differences from executor.map():
1. submit() returns a Future immediately (non-blocking).
2. as_completed() yields futures in FINISH order, not submission order.
3. Good for: progress reporting, early stopping, error handling.
4. ProcessPoolExecutor because training is CPU/GPU-bound.
5. Each "experiment" is independent → perfect task parallelism.
In real life: each train_experiment would call PyTorch training loop.
The 2-second sleep simulates that. With 4 workers, 8 experiments
finish in ~4s instead of ~16s.
-->

---

## `Future` Objects: What You Can Do

```python
future = executor.submit(train_experiment, config)

future.result()          # Block until done, return value (or raise exception)
future.done()            # True if finished
future.running()         # True if currently executing
future.cancel()          # Try to cancel (only works if not yet started)
future.exception()       # Get exception (None if success)
```

### Error handling pattern:

```python
with ProcessPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(train_experiment, c): c for c in configs}
    for future in as_completed(futures):
        config = futures[future]
        try:
            result = future.result()
            print(f"✓ {config} → loss={result['val_loss']:.3f}")
        except Exception as e:
            print(f"✗ {config} → FAILED: {e}")
```

> One experiment crashing doesn't kill the others. You collect what succeeded.

---

## `ProcessPoolExecutor`: Parallel Tokenization

### Scenario: Tokenize 8 documents with a pure-Python tokenizer.

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

def tokenize_python(text: str) -> list[int]:
    """Pure-Python char-level tokenizer. CPU-bound. GIL-bound."""
    vocab = {}
    tokens = []
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)
        tokens.append(vocab[word])
    return tokens

documents = ["transformer attention mechanism gradient descent " * 5000] * 8

# Threads: GIL serialises → NO speedup
with ThreadPoolExecutor(max_workers=4) as ex:
    t0 = time.perf_counter()
    results = list(ex.map(tokenize_python, documents))
    t_threads = time.perf_counter() - t0

# Processes: each has own GIL → ~4× speedup
with ProcessPoolExecutor(max_workers=4) as ex:
    t0 = time.perf_counter()
    results = list(ex.map(tokenize_python, documents))
    t_procs = time.perf_counter() - t0

print(f"Threads:   {t_threads:.2f} s")
print(f"Processes: {t_procs:.2f} s  ({t_threads/t_procs:.1f}× faster)")
```

> 📁 Full script: `class-examples/week04/04_parallel_tokenization.py`

<!-- note:
"How would you fix this without using processes?"
Answer 1: Use HuggingFace `tokenizers` library (Rust/C, releases GIL).
Answer 2: Use Numba @njit on the tokenizer (Week 2).
Answer 3: Use ProcessPoolExecutor (what we just did).
The point: you have options, but you must understand WHY threads fail here.
-->

---

## Batch Augmentation: Threads Work (NumPy Releases GIL)

### Scenario: Augment 32 images with NumPy operations.

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def augment_image(img: np.ndarray) -> np.ndarray:
    """Random flip + brightness jitter. All NumPy → GIL released."""
    if np.random.rand() > 0.5:
        img = img[:, ::-1, :]  # horizontal flip
    img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
    return img

batch = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(32)]

with ThreadPoolExecutor(max_workers=8) as ex:
    augmented = list(ex.map(augment_image, batch))
```

> Threads give ~4–6× speedup here because NumPy operations release the GIL.
> Same CPU-bound work, but done in C → threads parallelise it.

<!-- note:
 threads don't work for PYTHON CPU-bound code.
But NumPy operations are C code that releases the GIL.
So threads CAN parallelise NumPy-heavy augmentation.
The rule is NOT "threads = I/O, processes = CPU."
The rule is: "threads work when the GIL is released."
NumPy, BLAS, CUDA, HuggingFace tokenizers → GIL released → threads OK.
Pure Python loops → GIL held → need processes.
-->

---

## Choosing `max_workers`

| Executor | Workload | Recommended `max_workers` |
|----------|----------|:-------------------------:|
| `ThreadPoolExecutor` | LLM API calls, file I/O | 10–50+ |
| `ThreadPoolExecutor` | NumPy augmentation | 4–8 (= cores) |
| `ProcessPoolExecutor` | Tokenization, preprocessing | `os.cpu_count()` |
| `ProcessPoolExecutor` | Training experiments | `os.cpu_count()` or fewer |

```python
import os
cores = os.cpu_count()  # e.g., 8, 16
```

**Rules of thumb:**
- **Threads (I/O):** threads mostly *wait*. 50 threads waiting on network is fine.
- **Processes (CPU):** more than `cpu_count` → context-switch thrashing. Stay ≤ cores.
- **PyTorch DataLoader:** `num_workers=4–8` typical. More isn't always better (RAM).

---

## Common Pitfalls

| Mistake | Why it fails | Fix |
|---------|-------------|-----|
| Passing a lambda to `ProcessPoolExecutor` | Lambdas aren't picklable | Use a named `def` function |
| Sharing a model across processes | Processes don't share memory | Each process loads its own copy, or use threads |
| `max_workers=100` for processes | 100 processes thrash the OS | Use `os.cpu_count()` |
| Forgetting `with` statement | Zombie threads/processes | Always use `with executor:` |
| Passing a GPU tensor as argument | Can't pickle CUDA tensors | Pass CPU data, move to GPU inside worker |
| Benchmarking first call | Includes process pool creation | Warm up with a dummy task first |

```python
# ✗ This will FAIL:
with ProcessPoolExecutor() as ex:
    ex.submit(lambda x: x**2, 10)  # PicklingError!

# ✓ Fix:
def square(x): return x**2
with ProcessPoolExecutor() as ex:
    ex.submit(square, 10)
```

---

## Threads vs. Processes: Decision Summary

```text
Is your AI task I/O-bound?
(API calls, file loading, network, sleep)
    │
    ├── YES → ThreadPoolExecutor(max_workers=10–50)
    │
    └── NO (CPU-bound)
         │
         ├── Does it use NumPy / BLAS / CUDA / C extensions?
         │   ├── YES → ThreadPoolExecutor works (GIL released)
         │   └── NO (pure Python loop)
         │        │
         │        └── ProcessPoolExecutor(max_workers=cpu_count)
         │
         └── Is it a single large array op?
             └── Just use NumPy / Numba (Week 2). No executor needed.
```

> **This flowchart answers 95% of "threads or processes?" questions in AI.**

---

## Live Demo Summary: Run All Four

> 📁 `class-examples/week04/05_lab_concurrent_futures.py`

| # | AI Task | Executor | Expected |
|---|---------|----------|----------|
| 1 | Batch LLM inference (20 prompts) | `ThreadPoolExecutor(10)` | ~10× speedup |
| 2 | Tokenize 8 docs (pure Python) | `ProcessPoolExecutor(4)` | ~4× speedup |
| 3 | Tokenize 8 docs (pure Python) | `ThreadPoolExecutor(4)` | ~1× (GIL!) |
| 4 | Augment 32 images (NumPy) | `ThreadPoolExecutor(8)` | ~4–6× speedup |

Fill in the results table:

| Task | Serial | Threads | Processes | Best | Why? |
|------|-------:|--------:|----------:|------|------|
| LLM inference | ? | ? | — | ? | ? |
| Tokenize (Python) | ? | ? | ? | ? | ? |
| Augment (NumPy) | ? | ? | — | ? | ? |

---

## Key Takeaways – Parts I & II

1. **The GIL** serialises Python bytecode across threads.
   - I/O-bound (API calls, file loading) → **threads** (GIL released during wait).
   - CPU-bound pure Python → **processes** (each has own GIL).
   - CPU via NumPy/BLAS/CUDA → **threads** work (C code releases GIL).

2. **`concurrent.futures`** is the clean, modern API.
   - `executor.map()` for simple parallel-map patterns.
   - `executor.submit()` + `as_completed()` for progress tracking / error handling.

3. **`max_workers`:** threads can be many (I/O); processes ≤ `cpu_count`.

4. **In AI:**
   - LLM API calls → `ThreadPoolExecutor`
   - Tokenization / preprocessing → `ProcessPoolExecutor`
   - NumPy augmentation → `ThreadPoolExecutor`
   - `DataLoader(num_workers=N)` → processes internally (Week 9)

---

## Before We Continue (Break)

**Coming up after break:**
- Part III: Async / coroutines for massive I/O (LLM serving, agent swarms)
- Part IV: Pools as AI infrastructure (DataLoader, inference servers)
- Part V: Hands-on lab

**If you want to experiment during break:**
- Change `max_workers` in the LLM inference example. What happens with 1? With 50?
- Try `ProcessPoolExecutor` for the LLM inference task. Does it work? Is it faster/slower? Why?

---

<!-- _class: lead -->

# Part III
# Async / Coroutines for Massive AI I/O

---

## The Thread Limitation for Massive I/O

You have **500 prompts** to send to an LLM inference server.

| Approach | Memory | Threads | Practical? |
|----------|--------|:-------:|:----------:|
| 500 threads | 500 × ~8 MB stack = **~4 GB** | 500 | ❌ Wasteful |
| 50 async tasks | 50 × ~KB = **~50 KB** | **1** | ✅ Efficient |

> Threads are **heavy**. Each one needs a full stack (~1–8 MB).
> Coroutines are **lightweight**. They share one thread's stack.

**Coroutines = cooperative multitasking on a single thread.**
One thread handles thousands of concurrent I/O operations by
**switching** between them whenever one is waiting.

<!-- note:
"Coroutines provide concurrency, but they do not provide parallelism."
One thread. Many tasks. The thread switches between tasks when one
is waiting (e.g., for network). No context switch overhead. No stack
per task. This is perfect for I/O-bound AI workloads: LLM API calls,
agent tool queries, streaming responses.
-->

---

## Coroutines vs. Threads vs. Processes

| | Threads | Processes | Coroutines |
|---|---|---|---|
| **Parallelism?** | Concurrent (GIL!) | True parallel | Concurrent only |
| **Memory per unit** | ~1–8 MB (stack) | Full address space | ~KB |
| **Switching** | OS kernel (~µs) | OS kernel (~ms) | User-level (~ns) |
| **Max practical** | ~1 000 | ~100 | **~1 000 000** |
| **Best for** | I/O (< 100 tasks) | CPU-bound | I/O (> 100 tasks) |
| **Python tool** | `threading` | `multiprocessing` | `asyncio` |
| **AI example** | DataLoader workers | Tokenization | LLM serving, agents |

> From your notes: goroutines in Go use ~2 KB stack vs. ~1 MB for OS threads.
> Python coroutines are similar: ~KB per task vs. ~MB per thread.

---

## Generator-Based Coroutines

```python
def data_filter(pattern):
    """Coroutine: filters incoming data for a pattern."""
    print(f"Filter: {pattern}")
    while True:
        data = (yield)          # ⏸️ Pause; wait for data to be sent
        if pattern in data:
            print("Match:", data)

f = data_filter("urgent")
next(f)                          # ▶️ Prime the coroutine (advance to first yield)
f.send("urgent: GPU out of memory")   # 🔴 Prints match
f.send("info: training epoch 5 done") # 🟢 No output
```

- `yield` **pauses** the coroutine and returns control to the caller.
- `.send(data)` **resumes** the coroutine, passing `data` as the yield result.
- No threads. No locks. Cooperative scheduling.

> This is the foundation. `async`/`await` is the modern, cleaner syntax.

<!-- note:
  the "under the hood" mechanism.  The generator pattern helps to
understand WHAT await does: it's a yield point where the coroutine
suspends and lets the event loop run something else.
-->

---

## `async` / `await`: The Modern Syntax

```python
import asyncio

async def query_llm(prompt: str) -> str:
    """Simulate async call to inference server."""
    await asyncio.sleep(0.2)   # ⏸️ Non-blocking wait (GIL released)
    return f"Response to: '{prompt[:30]}'"

async def batch_inference(prompts: list[str]) -> list[str]:
    tasks = [query_llm(p) for p in prompts]
    return await asyncio.gather(*tasks)   # Run ALL concurrently

prompts = [f"Prompt {i}" for i in range(50)]
results = asyncio.run(batch_inference(prompts))
print(f"Got {len(results)} responses")
```

| Keyword | Meaning |
|---------|---------|
| `async def` | Defines a coroutine function |
| `await` | Suspends this coroutine; lets event loop run others |
| `asyncio.gather()` | Runs multiple coroutines concurrently |
| `asyncio.run()` | Entry point: creates event loop, runs top-level coroutine |

> 50 prompts, one thread, ~0.2 s total (not 10 s).

<!-- note:
Key insight: `await asyncio.sleep(0.2)` is NOT `time.sleep(0.2)`.
time.sleep blocks the THREAD. asyncio.sleep suspends the COROUTINE
and lets the event loop run other coroutines.
So 50 coroutines each "sleeping" 0.2s finish in ~0.2s total,
because they all sleep simultaneously.
This is the same principle as threads releasing the GIL during I/O,
but without the thread overhead.
-->

---

## Async LLM Client: Batch Inference

> 📁 `class-examples/week04/06_async_llm_client.py`

```python
import asyncio, time

async def query_inference_server(prompt: str, latency: float = 0.2) -> str:
    """Simulate async HTTP POST to vLLM / TGI server."""
    await asyncio.sleep(latency)  # non-blocking network wait
    return f"Generated: '{prompt[:25]}...'"

async def main():
    prompts = [f"Explain concept {i} in one sentence." for i in range(50)]

    t0 = time.perf_counter()
    tasks = [query_inference_server(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    print(f"50 prompts, 1 thread, {elapsed:.2f} s")
    print(f"Serial would be: {50 * 0.2:.1f} s")
    print(f"Speedup: {50 * 0.2 / elapsed:.0f}×")

asyncio.run(main())
```

```text
Output:
  50 prompts, 1 thread, 0.20 s
  Serial would be: 10.0 s
  Speedup: 50×
```

---

## Agent Swarm: Concurrent Tool Calls

> 📁 `class-examples/week04/07_agent_swarm.py`

### Scenario: 5 AI agents, each queries 3 tools concurrently.

```python
import asyncio, random

async def call_tool(agent_id: int, tool: str) -> str:
    """Simulate agent calling a tool (search, calculator, code_exec)."""
    await asyncio.sleep(random.uniform(0.1, 0.4))
    return f"Agent{agent_id} ← {tool}: result"

async def agent_loop(agent_id: int) -> list[str]:
    tools = ["web_search", "calculator", "code_executor"]
    tasks = [call_tool(agent_id, t) for t in tools]
    return await asyncio.gather(*tasks)

async def swarm():
    agents = [agent_loop(i) for i in range(5)]
    all_results = await asyncio.gather(*agents)
    for results in all_results:
        for r in results:
            print(f"  {r}")

asyncio.run(swarm())
# 15 concurrent async calls. 1 thread. ~0.4 s total.
```

> **In AI agents:** each agent queries tools (APIs, databases, code executors).
> Async lets one thread manage hundreds of agents simultaneously.

<!-- note:
This is a realistic AI agent pattern. In production:
- Each agent has a loop: observe → think → act → observe.
- The "act" step calls tools (web search, code execution, API calls).
- With async, you can run 100 agents on one thread.
- With threads, 100 agents = 100 threads = ~800 MB stack.
- Async: 100 agents = ~100 KB.
This is why frameworks like LangChain, AutoGen, CrewAI use async internally.
-->

---

## Async Streaming: Token-by-Token Generation

> 📁 `class-examples/week04/08_async_streaming.py`

```python
import asyncio

async def llm_stream(prompt: str):
    """Simulate streaming tokens from an LLM (like OpenAI stream API)."""
    tokens = f"Backpropagation computes gradients via the chain rule.".split()
    for token in tokens:
        await asyncio.sleep(0.05)  # generation latency per token
        yield token

async def consume_stream(prompt: str):
    print(f"Prompt: {prompt}\nResponse: ", end="", flush=True)
    count = 0
    async for token in llm_stream(prompt):
        print(token, end=" ", flush=True)
        count += 1
    print(f"\n[{count} tokens streamed]")

asyncio.run(consume_stream("What is backpropagation?"))
```

```text
Prompt: What is backpropagation?
Response: Backpropagation computes gradients via the chain rule.
[9 tokens streamed]
```

> `async for` + `async yield` = **async generator**.
> This is exactly how vLLM, TGI, and OpenAI streaming APIs work.

<!-- note:
The async generator pattern (async for + yield) is how streaming
LLM responses work in production. The server yields tokens one by one;
the client consumes them as they arrive. No need to wait for the full
response. This is the pattern behind:
- OpenAI's stream=True API
- vLLM's streaming endpoint
- HuggingFace TGI's streaming
-->

---

## Async vs. Threads: When to Use Which

| Criterion | Threads | Async |
|-----------|---------|-------|
| Number of concurrent tasks | < 100 | > 100 |
| Task type | I/O-bound (network, disk) | I/O-bound (network, API) |
| Memory concern | Moderate | Critical (1000s of tasks) |
| Code complexity | Simple (`ThreadPoolExecutor`) | Requires `async`/`await` |
| Library support | Universal | Growing (`aiohttp`, `httpx`) |
| AI example | DataLoader workers, file I/O | LLM serving, agent swarms |

> **Rule of thumb:**
> - < 50 concurrent I/O tasks → `ThreadPoolExecutor` (simpler)
> - > 100 concurrent I/O tasks → `asyncio` (scales better)
> - CPU-bound → `ProcessPoolExecutor` (neither threads nor async help)

---

## Async in the AI Ecosystem

| System | Uses async for |
|--------|---------------|
| **vLLM / TGI** | Handling 1000s of concurrent inference requests |
| **OpenAI API client** | Streaming responses, batch completions |
| **LangChain / AutoGen** | Agent tool calls, LLM queries |
| **Triton Inference Server** | Request queuing, batching |
| **Ray Serve** | Async model serving endpoints |
| **FastAPI** | Wrapping model inference in HTTP endpoints |

> You will encounter `async`/`await` in every AI deployment stack.
> Understanding it is not optional—it's infrastructure.

---

<!-- _class: lead -->

# Part IV
# Pools as AI Infrastructure

---

## Why Pools? (The Buffet Analogy)

> From your notes: think of a **buffet restaurant**.

```text
┌─────────────────────────────────────────────────────────────┐
│  BUFFET RESTAURANT (= Thread/Process Pool)                  │
│                                                             │
│  🍽️ Buffet Line (= Task Queue)                             │
│  [task1][task2][task3][task4][task5]...                     │
│       │                                                     │
│       ▼                                                     │
│  👨‍🍳 Chefs (= Worker Threads/Processes)                    │
│  Chef 1: picks task1, executes, picks next                  │
│  Chef 2: picks task2, executes, picks next                  │
│  Chef 3: picks task3, executes, picks next                  │
│  Chef 4: (idle, waiting for tasks)                          │
│                                                             │
│  When queue is empty → chefs rest (condition variable wait) │
│  When task arrives → wake one chef (condition signal)       │
└─────────────────────────────────────────────────────────────┘
```

- **Pre-created workers** → no per-task creation overhead.
- **Task queue** → decouples submission from execution.
- **Condition variable** → workers sleep when idle, wake when tasks arrive.

> `ThreadPoolExecutor` and `ProcessPoolExecutor` **are** thread/process pools.
> You've been using them all along.

---

## Pool Internals: What Happens on `submit()`

```python
with ThreadPoolExecutor(max_workers=4) as pool:
    future = pool.submit(query_llm, "What is attention?")
```

```text
1. pool.submit() wraps (query_llm, args) into a Task object
2. Task is placed on an internal Queue (thread-safe)
3. One of 4 worker threads picks it up (queue.get())
4. Worker executes query_llm("What is attention?")
5. Result is stored in the Future object
6. future.result() returns the value (or raises exception)
```

```text
Internal structure:
┌────────────────────────────────────────────────┐
│  ThreadPoolExecutor(max_workers=4)             │
│                                                │
│  Task Queue: [task1][task2][task3]...          │
│                                                │
│  Worker 0: ████████ (executing task1)          │
│  Worker 1: ████████ (executing task2)          │
│  Worker 2: ░░░░░░░░ (idle, waiting on queue)   │
│  Worker 3: ░░░░░░░░ (idle, waiting on queue)   │
│                                                │
│  Condition Variable: wakes idle workers        │
└────────────────────────────────────────────────┘
```

---

## Pool Sizing for AI Workloads

| Workload | Pool type | `max_workers` | Reasoning |
|----------|-----------|:-------------:|-----------|
| LLM API calls (I/O) | Thread | 10–50 | Threads mostly wait on network |
| File loading (I/O) | Thread | 8–16 | Disk I/O, some CPU for decode |
| Tokenization (CPU, Python) | Process | `cpu_count` | CPU-bound, GIL |
| Image augmentation (NumPy) | Thread | `cpu_count` | GIL released by NumPy |
| Training experiments | Process | `cpu_count` or fewer | CPU/GPU-bound |
| GPU inference | 1 per GPU | 1 (per GPU) | GPU is the bottleneck |

> From your notes:
> **CPU-bound:** `max_workers = cpu_count`
> **I/O-bound:** `max_workers = cpu_count × 5` (or more)

---

## DataLoader: A Process Pool + Queue

```python
loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

```text
┌─────────────────────────────────────────────────────────────┐
│  DataLoader Internals                                       │
│                                                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│  │Worker 0 │ │Worker 1 │ │Worker 2 │ │Worker 3 │         │
│  │(process)│ │(process)│ │(process)│ │(process)│         │
│  │         │ │         │ │         │ │         │         │
│  │ 1. Load │ │ 1. Load │ │ 1. Load │ │ 1. Load │         │
│  │    image│ │    image│ │    image│ │    image│         │
│  │ 2. Aug- │ │ 2. Aug- │ │ 2. Aug- │ │ 2. Aug- │         │
│  │    ment │ │    ment │ │    ment │ │    ment │         │
│  │ 3. Put  │ │ 3. Put  │ │ 3. Put  │ │ 3. Put  │         │
│  │  in q   │ │  in q   │ │  in q   │ │  in q   │         │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘         │
│       └───────────┴───────────┴───────────┘               │
│                         │                                   │
│                         ▼                                   │
│            ┌──────────────────────┐                        │
│            │  multiprocessing.Queue│ ← bounded buffer       │
│            └──────────┬───────────┘                        │
│                       │                                    │
│                       ▼                                    │
│            ┌──────────────────────┐                        │
│            │  Main process        │                        │
│            │  • Get batch         │                        │
│            │  • .to(device)       │                        │
│            │  • Forward/backward  │                        │
│            └──────────────────────┘                        │
│                       │                                    │
│                       ▼                                    │
│            ┌──────────────────────┐                        │
│            │  pin_memory thread   │ ← async CPU→GPU copy   │
│            └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

- `num_workers=4` → 4 **processes** (not threads! augmentation is CPU-bound).
- Each worker: load → augment → put into queue.
- Main process: get from queue → transfer to GPU → train.
- `pin_memory=True` → dedicated **thread** for async CPU→GPU transfer.
- `prefetch_factor=2` → each worker prefetches 2 batches ahead.

> **This is the producer-consumer pattern from Week 3**, applied to AI.
> Workers = producers. Training loop = consumer. Queue = bounded buffer.

---

## LLM Inference Server: Async + GPU Pool

```text
┌─────────────────────────────────────────────────────────────────┐
│  vLLM / TGI / Triton Inference Server                          │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Async Event Loop (1 thread)                           │     │
│  │  • Accepts 1000s of concurrent HTTP connections        │     │
│  │  • Parses requests, tokenizes prompts                  │     │
│  │  • Queues requests for GPU workers                     │     │
│  │  • Streams tokens back to clients                      │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                         │
│                       ▼ (request queue)                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  GPU Worker(s)                                         │     │
│  │  • Continuous batching (multiple requests per batch)   │     │
│  │  • KV-cache management (PagedAttention)                │     │
│  │  • Forward pass → generate tokens                      │     │
│  └────────────────────┬───────────────────────────────────┘     │
│                       │                                         │
│                       ▼ (token queue)                           │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Async Response Handler                                │     │
│  │  • Streams tokens to clients (SSE / WebSocket)         │     │
│  │  • Manages per-request state                           │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘

Async I/O (Part III) ←→ GPU Worker Pool (Part IV)
```

> **One async thread** handles 10 000 connections.
> **GPU workers** do the actual computation.
> This is why vLLM can serve 1000s of users on one GPU.

---

## Summary: Parallelism Mechanisms in an AI System

| Component | Mechanism | Why |
|-----------|-----------|-----|
| Data loading from disk | **Processes** (DataLoader workers) | CPU-bound augmentation |
| Data transfer to GPU | **Thread** (pin_memory) | I/O-bound (PCIe) |
| GPU forward/backward | **CUDA** (internal) | GPU parallelism |
| Gradient sync (DDP) | **NCCL** (not Python) | GPU-to-GPU communication |
| Logging / metrics | **Thread** or **async** | I/O-bound |
| LLM inference serving | **Async** event loop + GPU pool | Massive concurrent I/O |
| Agent tool calls | **Async** | Many concurrent API calls |
| Hyperparameter search | **Processes** (or Ray) | Independent CPU/GPU tasks |

> You now understand every row. The rest of the course (Weeks 5–13)
> builds on these foundations.

---

<!-- _class: lead -->

# Part V
# Hands-On Lab

---

## Lab: Four AI Workload Experiments

> 📁 `class-examples/week04/09_lab_all.py`

| # | Scenario | Tool | Expected |
|---|----------|------|----------|
| 1 | **LLM batch inference:** 30 prompts, 200ms each | Serial vs. Threads(10) vs. Async | Serial: 6s, Threads: ~0.6s, Async: ~0.2s |
| 2 | **Corpus tokenization:** 8 docs, pure Python | Threads(4) vs. Processes(4) | Threads ≈ Serial (GIL!), Processes ≈ 4× |
| 3 | **Batch augmentation:** 32 images, NumPy | Serial vs. Threads(8) | Threads give ~4× (GIL released) |
| 4 | **Agent swarm:** 10 agents × 3 tools each | Async (30 concurrent) | All finish in ~0.4s |

### Your task:
1. Run each experiment. Record timings.
2. Fill in the results table.
3. Answer the discussion questions.

---

## Lab: Results Table

Fill in during the lab:

| Experiment | Serial | Threads | Processes | Async | Best | Why? |
|------------|-------:|--------:|----------:|------:|------|------|
| LLM inference | ? | ? | — | ? | ? | ? |
| Tokenization | ? | ? | ? | — | ? | ? |
| Augmentation | ? | ? | — | — | ? | ? |
| Agent swarm | ? | — | — | ? | ? | ? |

---

## Lab: Discussion Questions

1. **Why did threads fail for tokenization but work for augmentation?**
   *(Hint: what does the GIL block? What does NumPy do?)*

2. **Why is async better than threads for the LLM inference workload?**
   *(Hint: memory per task, number of tasks, what "waiting" costs.)*

3. **Could you use `ProcessPoolExecutor` for LLM API calls? Would it work? Is it a good idea?**
   *(Hint: pickling overhead, process creation cost, I/O-bound nature.)*

4. **In a real training pipeline (DataLoader + GPU training + logging),
   which components use which mechanism? Draw the diagram.**

5. **Bonus:** Modify the agent swarm to use 50 agents × 5 tools.
   Does async still work? Try the same with threads. What happens?

---

## Lab: Answers 

| # | Key insight |
|---|-------------|
| 1 | Tokenization is pure Python (GIL held). Augmentation uses NumPy (GIL released). |
| 2 | Async: 1 thread, ~KB per task. Threads: ~8 MB stack each. For 50+ tasks, async wins on memory. Both give similar latency for I/O. |
| 3 | Yes, it works. But process creation + pickling overhead makes it slower than threads for I/O-bound tasks. Threads are simpler and sufficient. |
| 4 | DataLoader = processes. GPU training = CUDA. Logging = thread or async. pin_memory = thread. |
| 5 | Async handles 250 concurrent tasks easily. Threads would need 250 × 8 MB = 2 GB stack. |

---

<!-- _class: lead -->

# Part VI
# Wrap-Up

---

## Key Takeaways – Week 4

1. **The GIL** serialises Python bytecode across threads.
   - I/O-bound (API calls, file loading) → **threads** (GIL released during wait).
   - CPU-bound pure Python → **processes** (each has own GIL).
   - CPU via NumPy / BLAS / CUDA → **threads** work (C code releases GIL).

2. **`concurrent.futures`** is the default API.
   `ThreadPoolExecutor` for I/O. `ProcessPoolExecutor` for CPU.
   `map()` for simple patterns. `submit()` + `as_completed()` for control.

3. **`asyncio`** is for massive I/O concurrency.
   LLM serving, agent swarms, streaming. One thread, thousands of tasks.

4. **Pools are AI infrastructure.**
   DataLoader = process pool + queue. Inference server = async + GPU pool.
   You'll build on these in Weeks 9–13.

5. **Every AI system uses a mix:**
   processes (data loading) + threads (I/O, transfer) + async (serving)
   + CUDA (GPU compute) + NCCL (distributed sync).

---

## Decision Flowchart (Take This Home)

```text
What's the bottleneck in your AI task?
│
├── Waiting for network / API / disk?
│   ├── < 50 concurrent tasks → ThreadPoolExecutor
│   └── > 100 concurrent tasks → asyncio
│
├── CPU computation in pure Python?
│   └── ProcessPoolExecutor (max_workers = cpu_count)
│
├── CPU computation via NumPy / BLAS?
│   └── ThreadPoolExecutor (GIL released)
│
├── GPU computation?
│   └── PyTorch / CUDA handles it. Don't add Python threads.
│
└── Mixed pipeline (load + augment + train + serve)?
    └── DataLoader (processes) + pin_memory (thread) + GPU
        + async serving layer
        → Weeks 9–13
```

---

## Before Next Week

**Week 5:** Task parallelism at scale – **Dask & Ray**.
- Distributing work across many tasks / machines.
- Lighter week. More "when to use what" than deep implementation.

**Reading (optional, 15 min):**
- `asyncio` docs: https://docs.python.org/3/library/asyncio.html
- `concurrent.futures` docs: https://docs.python.org/3/library/concurrent.futures.html
- vLLM blog (continuous batching): https://blog.vllm.ai

**If you're curious:**
- Install `aiohttp` and modify the async LLM client to hit a real endpoint.
- Try `asyncio.Semaphore(5)` to limit concurrent API calls to 5.
- What happens if you use `ProcessPoolExecutor` for the async LLM task?

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank You

### Next week: Dask, Ray & Task Parallelism at Scale

*YZM 513 – İstanbul Medeniyet Üniversitesi – Fall 2026*


## Class-examples files for Parts III–V

### `class-examples/week04/06_async_llm_client.py`

```python

```

### `class-examples/week04/07_agent_swarm.py`

```python

```

### `class-examples/week04/08_async_streaming.py`

```python

```

### `class-examples/week04/09_lab_all.py`

```python

```

---

## Final repo structure for Week 4

```
week04/
├── 01_gil_llm_vs_tokenize.py       ← Part I: GIL demo
├── 02_batch_llm_inference.py        ← Part II: executor.map (LLM)
├── 03_hyperparam_sweep.py           ← Part II: submit + as_completed
├── 04_parallel_tokenization.py      ← Part II: threads vs processes
├── 05_batch_augmentation.py         ← Part II: NumPy + threads
├── 06_async_llm_client.py           ← Part III: async batch inference
├── 07_agent_swarm.py                ← Part III: agent swarm
├── 08_async_streaming.py            ← Part III: token streaming
├── 09_lab_all.py                    ← Part V: combined lab
```

