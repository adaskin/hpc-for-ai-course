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

### Week 5: Beyond One Machine
### Decomposition, Coordination, and the Cost of Distribution

**İstanbul Medeniyet Üniversitesi**
AI Engineering M.Sc. – Fall 2026

Ammar Daşkın

---

## The Week in One Sentence

> Every parallel system — threads, MPI, Dask, Ray, GPU kernels, DDP —
> answers the same four questions.
> The answers change. The questions don't.

**The four questions:**

| # | Question | Week 3–4 answer | This week's answer |
|---|----------|----------------|-------------------|
| 1 | How do we **split** the work? | Chunk an array / split documents | Partitioning strategies |
| 2 | Who **does** what? | Thread pool / process pool | Schedulers, work-stealing |
| 3 | What **moves**, and what does it cost? | Shared memory (free) / pickle (expensive) | Object stores, networks |
| 4 | Who **waits** for whom? | Locks, barriers, queues | Task graphs, all-reduce |

> You already answered these for one machine. Today: the same questions, bigger scale.
> At cluster scale a fifth appears: **what if a worker dies?**

---

## Positioning in the Course

```text
Week 1–2:  One core, made fast     (vectorise, JIT, BLAS)
Week 3–4:  One machine, many cores (threads, processes, async, pools)
Week 5:    Many machines           ← TODAY
Week 6–8:  One GPU, many cores     (SIMT, CUDA, memory hierarchy)
Week 9:    Feeding the GPU         (DataLoader pipelines)
Week 10–11: Many GPUs              (DDP, model parallelism)
Week 12–13: Making it efficient    (mixed precision, torch.compile)
```

> Week 5 fills the "many machines" slot. But the *concepts* are the same
> ones you'll see in Week 10 (DDP = partitioning + all-reduce + barriers).

---

<!-- _class: lead -->

# Part I
# The Universal Questions of Parallelism

---

## Question 1: How Do We Split the Work?

From Week 3, you know two fundamental forms:

**Data parallelism:** same operation, different data.
- Matrix addition: each element independent.
- Tokenizing a corpus: each document independent.

**Task parallelism:** different operations, same or different data.
- Pipeline: load → preprocess → train → log.
- Hyperparameter sweep: 8 different experiments.

![w:600](img/w05-why-partition.svg)

> At any scale — 2 threads or 200 machines — you must choose
> *what* to split and *where* to cut.

---

## Question 2: Who Does What? (Assignment)

From Week 4, you saw the **buffet analogy**:

- **Static split:** "You do docs 0–999, you do docs 1000–1999."
  Simple. Breaks if one chunk is 10× slower.
- **Shared queue (pull-based):** workers grab the next task when free.
  This is what `ProcessPoolExecutor.map` does. Good load balancing.
- **Work-stealing:** idle workers steal queued tasks from busy ones.
  Dask's distributed scheduler does this. Ray's per-node schedulers
  send tasks to nodes that have free resources and the data.

![bg right:45% 95%](img/w05-scheduling.svg)

> The principle is identical at any scale.
> Only the mechanism changes (OS scheduler → framework scheduler → cluster manager).

---

## Question 3: What Moves, and What Does It Cost?

From Week 3: threads share memory → communication is "free" (just read the variable).
From Week 4: processes don't share → pickle/unpickle.

**This week:** workers on different machines don't share memory at all.

![bg right:45% 95%](img/w05-cost-ladder.svg)

| Where | Cost |
|-------|------|
| Same thread / shared heap | ~ns |
| Other process, same machine (pickle or shared memory) | ~µs–ms |
| Other machine (network: TCP, gRPC) | ~ms |

> The further the data must travel, the more it dominates.

---

## Question 4: Who Waits for Whom? (Synchronization)

From Week 3: locks, semaphores, barriers, condition variables.

At distributed scale, the same patterns appear under different names:

| Week 3 concept | Distributed equivalent | AI example |
|----------------|----------------------|------------|
| Lock / Mutex | Distributed lock (rare, expensive) | — |
| Barrier | Synchronous phase boundary | DDP: all GPUs finish backward before step |
| Condition variable | Event / future resolution | `ray.get()`, `dask.compute()` |
| Producer-consumer queue | Task queue / object store | DataLoader, Ray object store |
| Semaphore | Resource limiter | `num_gpus=1` in Ray |

> You already know the patterns. The vocabulary changes; the logic doesn't.

---

## New at Cluster Scale: What If a Worker Dies?

One machine: if it dies, everything dies together. Nobody is left to care.

> 100 machines × a 10-hour job = 1,000 machine-hours.
> If a machine fails once per 1,000 hours, expect **about one failure per job**.

| Strategy | Idea | Works well when |
|----------|------|-----------------|
| **Retry** | run the lost task again | the task is a pure function |
| **Recompute** | rebuild lost data from its recipe (Spark's *lineage*) | the graph was recorded |
| **Checkpoint** | save progress to storage | the state is expensive to rebuild |

> **Stateless** tasks are easy to retry. **Stateful** workers (an actor, a training
> process) lose their state when they die, so they must checkpoint.

---

<!-- _class: lead -->

# Part II
# The Cost Model: Why Distribution Is Hard

---

## The Overhead Hierarchy

Every parallel operation pays:

```text
Task creation:     ~µs (threads) to ~ms (processes, remote tasks)
Data serialization: ~ms per MB (fast for NumPy arrays, slow for Python objects)
Data transfer:     ~ms per message (network)
Synchronization:   ~µs (local lock) to ~ms (distributed barrier)
Result collection:  ~µs per result × number of results
```

**A simple cost model:**

$$T_{parallel} = T_{overhead} + T_{communication} + \frac{T_{compute}}{N}$$

Distribution helps only when the saving, $T_{compute}\left(1-\frac1N\right)$, exceeds $T_{overhead} + T_{communication}$.

---

## Amdahl's Law with Communication

Standard:
$$S(N) = \frac{1}{f_{serial} + \frac{1 - f_{serial}}{N}}$$

With communication:
$$S(N) = \frac{1}{f_{serial} + f_{comm} + \frac{1 - f_{serial} - f_{comm}}{N}}$$

- $f_{serial}$: the reduce step, the narrow waist, the final merge.
- $f_{comm}$: task overhead + serialization + transfer, as a fraction of the total.
- Ceiling: as $N \to \infty$, $S \to \dfrac{1}{f_{serial} + f_{comm}}$. If they sum to 0.1, **no number of workers gives more than 10×**.
- If $f_{comm}$ **grows** with $N$ (more workers → more messages), speedup can **decrease**.

> Next question: what controls $f_{serial}$ and $f_{comm}$? The **size of a task**.

---

## Granularity: The Dial Behind the f's

> **Granularity** = how much work is in one task.
> With $D$ documents and $g$ documents per task, there are $D/g$ tasks.

| | Too fine ($g$ small) | Too coarse ($g$ large) |
|---|---|---|
| **Tasks** | thousands | fewer than workers |
| **$f_{comm}$** (cost per task × #tasks) | large | small |
| **$f_{serial}$** (reduce over all partials) | large | small |
| **Workers busy?** | yes, but mostly on overhead | no: idle workers, stragglers |
| **Memory** | fine | a task may not fit |

**The sweet spot:** each task computes much longer than it costs to create, schedule, serialize and collect.

> This one dial explains more distributed performance problems than any other:
> Dask chunks, Ray tasks, DataLoader batches, GPU kernel launches, DDP gradient buckets.

---

## Measured: Granularity Through Amdahl's Lens

![w:820](img/w05-granularity.svg)

- Same work, same 2 workers. Only the **task size** changes.
- **Left:** the orange and grey parts are $f_{comm}$ and $f_{serial}$. They shrink as tasks grow.
- **Right:** speedup against the **ceiling of 2×**. Small tasks are *slower than not parallelising at all*.

> 📁 `class-examples/week05/01_granularity_sweep.py` (also times the serial version)

---

## Reading the Numbers With Amdahl

Serial version: **0.44 s**. Ceiling with 2 workers: **2×**.

| docs/task | tasks | total | speedup $S$ | effective $f$ |
|----------:|------:|------:|------------:|--------------:|
| 1 | 2000 | 1.82 s | 0.24× | 7.4 |
| 10 | 200 | 0.62 s | 0.70× | 1.8 |
| **100** | **20** | **0.29 s** | **1.52×** | **0.32** |
| 1000 | 2 | 0.32 s | 1.36× | 0.47 |

$f \approx \dfrac{1/S - 1/N}{1 - 1/N}$: the same estimator you use in the lab.
**$f > 1$ means the overhead costs more than the whole job.**

- Best: ~100 docs/task. 1.5× of the 2× ceiling; the rest is pool start-up and pickling.
- 1000 docs/task is slightly worse: only 2 tasks, each ships ~12 MB, and no slack if one worker lags *(likely cause; not isolated)*.

<!-- note:
Ask first: "2000 tasks on 2 workers: faster or slower than one process?" Most students say faster. It is 4x SLOWER.
f_eff is the lab's estimator applied to each row. Values above 1 are not real fractions: they say overhead exceeds the entire serial job.
-->

---

<!-- _class: lead -->

# Part III
# Decomposition: How to Cut

---

## Data Partitioning Strategies

| Strategy | Cut by | Choose when | AI example |
|----------|--------|-------------|------------|
| **Block / chunk** | Contiguous blocks | Arrays, matrices, images | Embedding matrix rows |
| **Row / document** | Whole records | Each record processed independently | Corpus tokenization |
| **Hash** | Hash of a key | Equal keys must meet (joins, groupby) | RAG store by doc ID |
| **Range** | Value range | Sorted data, range queries | Training shards by index |

> The right cut depends on what the **next operation** needs.
> If the next op is local to each piece → any cut works.
> If the next op needs to see all pieces → you'll pay a reduce or shuffle.

---

## The Shuffle Problem

**Good cut:** documents partitioned by ID, next op is "tokenize each doc."
→ Each piece is independent. No data moves. Embarrassingly parallel.

**Bad cut:** documents partitioned by date, next op is "group by author."
→ Each author's documents are scattered across all pieces.
→ Rows must **move** between workers to co-locate.

This is a **shuffle**: the most expensive operation in distributed computing.

> **Design principle:** choose your partitioning to match your access pattern.
> A column mean over row-partitioned data is a cheap **reduce** (each piece sends sum, count).
> A groupby over the wrong partition is an expensive **shuffle**.

---

## Task Decomposition: Pipeline vs. Data-Parallel vs. Task-Parallel

From Week 3's matrix examples:

| Pattern | Structure | Example |
|---------|-----------|---------|
| **Pipeline** | Stage 1 → Stage 2 → Stage 3 | DataLoader: load → augment → transfer |
| **Data-parallel** | Same op on N chunks | Batch of images through one layer |
| **Task-parallel** | N different ops simultaneously | Hyperparameter sweep: 8 configs |

Most real AI systems **combine** these:
- DataLoader = pipeline (stages) + data-parallel (multiple workers)
- DDP = data-parallel (shards) + synchronous barrier (gradient sync)
- Inference server = task-parallel (requests) + pipeline (prefill → decode)

---

<!-- _class: lead -->

# Part IV
# Coordination: Task Graphs and Communication

---

## The Task Graph: What Depends on What

- **Node** = a unit of work (a task)
- **Edge** = data dependency (output of A feeds input of B)
- **Width** = max tasks that can run simultaneously
- **Critical path** = longest dependency chain = minimum possible time

> If width = 1, you have a serial pipeline. No parallelism possible.
> The **narrow waist** is where width drops to 1. That's where Amdahl bites.

---

## Our Running Example: Document Frequency → IDF

![w:800](img/w05-dfidf-graph.svg)

- **Width 4** at the start (count each chunk) and end (weight each chunk).
- **Width 1** in the middle (merge all → compute IDF). This is the narrow waist.
- **Critical path:** count → merge → merge all → IDF → weight = 5 steps.
- **Total tasks:** 12. But minimum time = 5 steps (not 12/4).

> This is the heart of Project 1's TF-IDF search. The same graph appears
> in Dask (Part VI) and Ray (Part VI), built differently but identical in structure.

---

## Static vs. Dynamic Graphs

| | Static graph | Dynamic graph |
|---|---|---|
| **Built** | Before execution | As the program runs |
| **You can** | Inspect, optimise, visualise | Branch on results, loop until done |
| **Cost** | Rigid shape | No global optimisation |
| **Examples** | Dask collections, `dask.delayed`, Spark | Ray tasks, Dask futures |

> **Connection to PyTorch:** eager mode = dynamic (each op runs immediately).
> `torch.compile` (Week 13) = captures a static graph from eager code.
> Same concept, different domain.

---

## Four Communication Patterns

| Pattern | Meaning | Week 3 analog | AI example |
|---------|---------|---------------|------------|
| **Broadcast** | Same data to all workers | — | Send IDF weights to all chunks |
| **Reduce / All-reduce** | Combine results from all | Barrier + merge | DDP gradient sync (Week 10) |
| **Shuffle** | Redistribute by key | — | Repartition dataset by length |
| **Point-to-point** | Send between specific workers | Producer → consumer | Pipeline stages (Week 11) |

> In Week 3, you used locks and queues for coordination.
> At distributed scale, these four patterns replace them.
> The **all-reduce** is the most important one for AI (Week 10: DDP).

---

## All-Reduce: Reduce, Then Everyone Gets the Result

```text
worker 0: g0 ─┐                          ┌─► g0+g1+g2+g3
worker 1: g1 ─┤   reduce (sum)           ├─► g0+g1+g2+g3
worker 2: g2 ─┤   + broadcast the sum    ├─► g0+g1+g2+g3
worker 3: g3 ─┘                          └─► g0+g1+g2+g3
```

- **DDP (Week 10):** each GPU computes gradients on its own data shard; all-reduce gives **every** GPU the same summed gradient before the optimiser step.
- A **ring all-reduce** passes slices around a ring: each worker sends about 2× the data size in total, almost **independent of the number of workers**.
- It is also a **barrier**: nobody continues until everyone has the result.

---

## The Cost of Communication

**Rule of thumb:**
> Move computation to data when data is big.
> Move data to computation when computation is big and data is small.

| Scenario | What to do |
|----------|-----------|
| 500 GB corpus, small operation per doc | Send the code to the data (partition locally) |
| Small query vector, huge index | Send the query to where the index lives |
| Model weights needed by all workers | Broadcast once, not per-task |
| Gradients from all GPUs | All-reduce (Week 10) |

> This is why `ray.put()` and `client.scatter()` exist:
> put big read-only data somewhere once, pass a reference.

---

<!-- _class: lead -->

# Part V
# The Landscape of Systems

---

## The Spectrum of Parallel Tools

```text
┌────────────────────────────────────────────────────────────────────┐
│  Scope          Tool                    You manage                 │
├────────────────────────────────────────────────────────────────────┤
│  One core       NumPy / Numba           Nothing (BLAS does it)     │
│  One machine    threading / asyncio     Thread pool, queues        │
│  One machine    multiprocessing         Process pool, pickle       │
│  One machine    Dask (local)            Chunk sizes, scheduler     │
│  Cluster        Dask distributed        Cluster setup              │
│  Cluster        Ray                     Cluster setup, actors      │
│  Cluster        Spark                   Cluster, partitions        │
│  Cluster        MPI                     Everything by hand         │
│  GPU            CUDA / PyTorch          Kernel design              │
│  Multi-GPU      DDP / DeepSpeed         Gradient sync strategy     │
└────────────────────────────────────────────────────────────────────┘
```

> Each row answers the same four questions differently.
> The further down you go, the more the framework manages for you.

---

## A Brief History (Vocabulary)

| Year | System | Key idea | Legacy |
|------|--------|----------|--------|
| 1994 | **MPI** | Explicit messages between ranks | Still used for HPC numerics |
| 2004 | **MapReduce** | map → shuffle → reduce; fault tolerance by re-execution | The pattern every framework copies |
| 2012 | **Spark** | In-memory (RDDs), lineage-based recovery | Industry standard for ETL/analytics |
| 2015 | **Dask** | "NumPy/pandas but bigger", PyData ecosystem | Natural fit for Python data science |
| 2018 | **Ray** | Tasks + actors, built for ML | Growing fast in AI infrastructure |

> You will meet **Spark** in industry. You will use **Dask/Ray** in this course.
> The concepts (partition, schedule, communicate, recover) are the same in all of them.

---

## Cluster, OpenMP, MPI: Three Words You Will Meet

![w:620](img/w05-memory-models.svg)

- **Cluster:** nodes (each with **its own** RAM) + a network + a job scheduler (Slurm).
- **OpenMP:** threads sharing one node's memory, via a compiler pragma:
  `#pragma omp parallel for reduction(+:sum)`
- **MPI:** processes with separate memory that **send messages** (a library standard).

---

## MPI: The Manual Baseline (2 minutes)

```c
MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // who am I?
MPI_Comm_size(MPI_COMM_WORLD, &size);   // how many of us?

for (long i = rank; i < N; i += size)   // my share
    local += 1.0 / (i + 1);

MPI_Reduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
```

**What MPI makes you do by hand:**
- Split the work (`i = rank; i += size`)
- Combine results (`MPI_Reduce`)
- Handle failure (if one rank dies, the whole job stops)

> Dask and Ray automate exactly this bookkeeping.
> That's their value proposition.


---

## What Are Dask and Ray For?

| | **Dask** | **Ray** |
|---|---|---|
| **What it is** | A Python library that scales NumPy, pandas and plain Python | A Python framework that runs functions and objects on many workers |
| **You write** | Code that looks like NumPy / pandas | `@ray.remote` functions and classes |
| **Used for** | Data larger than RAM, tables and arrays, scientific data, ETL | Many independent runs, hyper-parameter search, serving, distributed training |

> Neither is an arithmetic library. Both are **coordinators** for Python code.

---

## Dask vs. Ray

| | **Dask** | **Ray** |
|---|---|---|
| **You think in** | Chunked arrays / tables / bags | Function calls and objects |
| **Mental model** | "NumPy / pandas, but bigger" | "`f()`, but on another worker" |
| **Graph** | Built from collection code (static) or futures (dynamic) | Built as program runs (dynamic) |
| **Data sharing** | `client.scatter()` | `ray.put()` → object store |
| **Stateful workers** | Possible, not primary | Actors (core feature) |
| **Fault tolerance** | Re-run lost tasks | Retry tasks; restart actors |
| **Strongest at** | Data bigger than RAM | Many independent tasks, ML pipelines |
| **Grew from** | PyData (NumPy/pandas maintainers) | ML systems research (Berkeley) |

> They overlap. For Project 1, either could work. The point is to know *why* you'd pick one.

---

## How They Talk, and Where the Threads Are

| | Between processes | Workers and memory |
|---|---|---|
| **Dask** | **TCP** by default; UCX optional for fast interconnects | Separate worker **processes**, each may run several **threads**; no memory shared **between** workers |
| **Ray** | **gRPC** for control; **shared-memory object store** for data | Separate worker **processes**; workers on one node read large NumPy arrays **zero-copy** from the store |

> Both use their **own** protocols: not MPI, not OpenMP.
> Between machines, data is always serialised and sent.

<!-- note:
dask-mpi / dask-jobqueue only help LAUNCH Dask workers inside an MPI / Slurm job; the data still travels over Dask's own comms.
The Ray object store is per node: moving an object to another node is a network transfer.
-->

---

## The Stack: Who Does the Arithmetic?

![w:750](img/w05-stack.svg)

> **Dask and Ray do NOT do the arithmetic.**
> They decide *what runs where* and *move data*.
> NumPy / BLAS / your code does the actual computation.

---

## A Hidden Trap: Threads Inside Workers

NumPy calls BLAS, which spawns its own threads.
4 workers × 4 BLAS threads = 16 threads on 4 cores → **oversubscription**.

Fix: `workers × BLAS_threads ≤ cores`. Set `OMP_NUM_THREADS=1` per worker.

```text
BLAS threads per worker | total threads | time   (2 cores, our run)
          1             |       2       | 0.73 s
          4             |       8       | 0.93 s
```

> 📁 `class-examples/week05/09_blas_oversubscription.py`

---

<!-- _class: lead -->

# Part VI
# Dask and Ray in Practice
# (Same Problem, Two Styles)

---

## The Workload: DF → IDF

> The same problem, solved with Dask and with Ray (and compared with serial and ProcessPool in the lab).

```text
1. Split corpus into chunks
2. Count document frequency per chunk        (parallel, width = #chunks)
3. Merge all counts                          (reduce, narrow waist)
4. Compute IDF from merged counts            (serial, tiny)
5. Weight each chunk using IDF               (parallel, width = #chunks)
```

This is the graph from Part IV. Now we implement it.

---

## Dask: Build the Graph, Then Run It

```python
from dask import delayed

counts  = [delayed(df_chunk)(c) for c in chunks]       # width N
m1      = delayed(merge)(counts[0], counts[1])          # tree reduce (4 chunks shown)
m2      = delayed(merge)(counts[2], counts[3])
total   = delayed(merge)(m1, m2)                        # narrow waist
idf     = delayed(make_idf)(total, n_docs)
weights = [delayed(weight_chunk)(c, idf) for c in chunks]  # width N again

# Nothing has run yet. Visualise first:
dask.visualize(*weights, filename="df_idf_graph.png")

# Then execute:
results = dask.compute(*weights)
```

- **Static graph:** built before execution. You can inspect it.
- **Lazy:** nothing runs until `.compute()`.
- **Scheduler:** threads by default (GIL!). Use `scheduler="processes"` for pure Python.

> 📁 `class-examples/week05/02_dask_delayed_graph.py`

---

## Ray: Build the Graph While Running

```python
import ray

@ray.remote
def df_chunk(docs): ...

@ray.remote
def merge(a, b): ...

chunk_refs = [ray.put(c) for c in chunks]          # store once
counts     = [df_chunk.remote(c) for c in chunk_refs]  # width N
total_df   = tree_reduce(counts)                    # merge futures pairwise
idf_ref    = make_idf.remote(total_df, n_docs)      # narrow waist
weights    = [weight_chunk.remote(c, idf_ref) for c in chunk_refs]
result     = sum(ray.get(weights))
```

- **Dynamic graph:** built by ordinary Python loops while running.
- **Eager submission:** `.remote()` returns immediately.
- **Object store:** `ray.put()` stores data once; pass references.

> 📁 `class-examples/week05/16_ray_df_idf.py`

---

## Same Graph, Different Styles

| | Dask `delayed` (local) | Dask distributed | Ray tasks |
|---|---|---|---|
| Graph built | Before execution | Before execution | While running |
| Can you visualise it? | Yes: `dask.visualize()` | Yes (dashboard) | No: you see it execute |
| Branch on a result? | Awkward | Awkward | Natural |
| Shared data | `client.scatter()` | `client.scatter()` | `ray.put()` |
| Default scheduler | **Threads (GIL!)** | **Processes** | **Processes** |
| Best for | Array/table pipelines | Out-of-core, clusters | Many independent tasks, ML |

> The **concepts are identical**: tasks, dependencies, narrow waist, broadcast.
> Only the *style* and the *default scheduler* differ.
> Dask `delayed` on threads will hit the GIL for pure-Python tasks.
> Switch to `scheduler="processes"` or use `distributed` to fix it.

<!-- note:
This is why example 02_dask_delayed_graph says "pure-Python tasks on threads: the GIL limits the speedup."
And why 08_dask_distributed_futures uses LocalCluster(n_workers=2, threads_per_worker=1).
And why Ray tasks do not share a GIL: they run in separate worker processes (reused, not one per task).
The scheduler choice is the SAME decision as Week 4: threads for I/O / NumPy,
processes for pure Python. Nothing new.
-->

---

## The Graph Is Just the Recipe

We've built the same DF → IDF graph in both tools.
The graph tells you **what** runs and **in what order**.

But the graph doesn't move data.

```text
   Graph structure:        Data reality:
   ┌─────────────┐         ┌─────────────────────────────────────┐
   │ count chunk1│         │ 20 chunks × ~300 KB each            │
   │ count chunk2│         │ must arrive at 20 different workers │
   │ count chunk3│         │                                     │
   │     ...     │         │ 1 IDF dict × ~100 KB, grows w/ vocab│
   │ weight all  │         │ must arrive at ALL 20 workers       │
   └─────────────┘         └─────────────────────────────────────┘
```

---

## Free to Draw, Expensive to Move

> **The graph is free to draw. The data is expensive to move.**
>
> In our DF → IDF example:
> - The 20 chunks are used **twice** (count step AND weight step).
> - The IDF dict is needed by **every** weight task.
>
> If we pass them by value, they get serialized with every task call.
> 20 chunks × 2 uses = 40 serializations. Plus 20 copies of the IDF dict.
>
> **The fix:** store once, pass a reference. Both tools have this.

<!-- note:
This is the bridge from "how to express the computation" to
"how to feed it efficiently." The graph was the STRUCTURE lesson.
Data movement is the COST lesson. 
-->


---


## Object Stores: Pass References, Not Data

```python
# BAD: 20 MB array pickled with EVERY task
futures = [score.remote(matrix, i) for i in range(20)]

# GOOD: serialize once, pass a reference
ref = ray.put(matrix)
futures = [score.remote(ref, i) for i in range(20)]
```

| | By value | By reference (`ray.put`) |
|---|---|---|
| Serializations | 20 × 20 MB = 400 MB | 1 × 20 MB |
| Time (your run) | ? | ? |

> Same idea as Week 4's `process_pool_search.py` fix.
> Same idea as Dask's `client.scatter(big, broadcast=True)`.
> **Principle:** put big read-only data somewhere once, pass a reference.

> 📁 `class-examples/week05/11_ray_object_store.py`  (prints both times)

---

## Stateful vs. Stateless: Tasks vs. Actors

| | Stateless task | Stateful actor |
|---|---|---|
| Setup cost | Paid **every** call | Paid **once** per actor |
| State between calls | None | Kept in memory |
| Fault tolerance | Easy: just retry | Hard: state is lost on crash |
| Scaling | Trivial (any worker) | Harder (actor pinned to one process) |
| AI example | Tokenize a document | LLM model loaded once, serves many queries |

```python
@ray.remote
class Predictor:
    def __init__(self):
        self.W = load_model()       # 1 second, paid ONCE
    def predict(self, x):
        return float((self.W @ x).sum())
```

> 📁 `class-examples/week05/14_ray_actors.py`

---

## Granularity in Practice (Both Tools)

| docs/task | tasks | ProcessPool (Week 4) | Dask Bag | Ray |
|----------:|------:|---------------------:|---------:|----:|
| 1 | 2000 | 1.82 s | ? | ? |
| 10 | 200 | 0.62 s | ? | ? |
| **100** | **20** | **0.29 s** | ? | ? |
| 1000 | 2 | 0.32 s | ? | ? |

> **Predict:** does the sweet spot move with the framework?
> The bottleneck is the same: per-task overhead + a serial reduce.
> Frameworks change the ergonomics; the physics stays.

> 📁 `01_granularity_sweep.py` (ProcessPool, Part II) · `05_dask_bag_df_idf.py` · `13_ray_granularity.py`
> *(ProcessPool column: our 2-core sandbox run.)*

---

## Dask Extras (Brief)

**Dask Array:** chunked NumPy. Lazy. Out-of-core.
```python
x = da.ones((100_000, 5000), chunks=(10_000, 5000))  # 4 GB, never in RAM at once
(x * 2 + 1).sum().compute()
```

**Dask DataFrame:** chunked pandas. Same API.
```python
ddf = dd.read_csv("data/*.csv")
ddf.groupby("author")["length"].mean().compute()
```

**Schedulers:** threads (NumPy ops, GIL released) vs. processes (pure Python).
Same GIL logic from Week 4.

> 📁 `class-examples/week05/03_dask_array_chunks.py` (optional demo)

---

<!-- _class: lead -->

# Part VII
# Choosing: Should You Distribute at All?

---

## The Decision Tree

```text
Which wall are you hitting?

MEMORY: data > RAM
├── array / table          → Dask Array / DataFrame
├── bag of text / records  → Dask Bag, or Ray tasks
└── more than one machine  → Dask distributed

COMPUTE / TIME: data fits in RAM
├── one array operation              → NumPy / BLAS (Week 2). Done.
├── pure-Python CPU-bound            → ProcessPoolExecutor (Week 4). Done.
├── I/O-bound                        → threads or asyncio (Week 4). Done.
└── thousands of independent tasks,
    or workers that must keep state  → Ray (a cluster, if one machine is not enough)
```

> **For Project 1 (5k–20k docs, 2-core Colab):**
> The honest answer is often "plain NumPy or a process pool wins."
> That is a valid, well-justified result.

---

## Project 1 Phase 3: Checklist

When writing your "Why not the others?" section:

1. **Does the data fit in RAM?** If yes, Dask's main benefit disappears.
2. **Work per task vs. overhead?** Use your granularity curve.
3. **How much data moves per task?** Can it be stored once? (`ray.put` / scatter)
4. **Is the serial part (reduce / top-k merge) small?**
5. **Workers × BLAS threads ≤ cores?**

> If you can't justify distribution with numbers, don't distribute.
> A well-reasoned "I chose not to use Dask/Ray because..." scores full marks.

---

## Forward Map: Where These Concepts Reappear

| Week 5 concept | Where it reappears |
|----------------|-------------------|
| Task graph / DAG | `torch.compile` graph capture (Week 13) |
| Partitioning / chunking | DataLoader sharding (Week 9), DDP data shards (Week 10) |
| All-reduce | DDP gradient sync via NCCL (Week 10) |
| Point-to-point | Pipeline parallelism stages (Week 11) |
| Broadcast | Model weights to all GPUs at init (Week 10) |
| Checkpoint / retry | Saving training state (Weeks 9–11) |
| Out-of-core / streaming | WebDataset for massive datasets (Week 9) |
| Granularity | GPU kernel launch overhead (Week 6), DataLoader batch size (Week 9) |

> You now have the vocabulary. When you read a paper on Megatron-LM,
> DeepSpeed, or vLLM, you'll recognise the graph, the partitioning,
> the communication pattern, and the failure strategy.

---

## Key Takeaways

1. **Every parallel system answers four questions:** split, assign, move, synchronise.
   The answers change with scale; the questions don't.

2. **Distribution adds cost.** Only use it when cheaper options (vectorise, JIT,
   local pools) have run out.

3. **Granularity is the master dial.** Too fine → overhead dominates.
   Too coarse → no parallelism. Measure, don't guess.

4. **Moving data is usually the real cost.** Store once, pass references.
   Design partitions to minimise shuffles.

5. **Dask and Ray coordinate; NumPy/BLAS compute.** Watch workers × threads.

6. **At scale, failure is normal.** Retry stateless tasks; checkpoint stateful workers.

7. **Often the right answer is: don't distribute.**

---

<!-- _class: lead -->

# Lab / Homework

---

## Lab: DF → IDF, Three Ways

Run on ≥ 5,000 documents (your Project 1 corpus is fine).

| Version | File | Time | Speedup |
|---------|------|------|---------|
| Serial | `corpus.py` helpers | ? | 1× |
| ProcessPool (Week 4) | reuse | ? | ? |
| Dask `delayed` | `02_dask_delayed_graph.py` | ? | ? |
| Ray | `16_ray_df_idf.py` | ? | ? |

**Answer these:**
1. Draw the task graph. Width? Critical path? Where's the narrow waist?
2. What chunk size did you choose? Evidence?
3. What data moved, roughly how many bytes, and what did it cost?
4. Estimate the serial fraction as on the sweep slide: $f \approx \frac{1/S - 1/N}{1 - 1/N}$ *(an upper bound: it also contains start-up, pickling and the reduce)*

> **Expect the plateau:** Colab has 2 cores. 4 workers won't beat 2.
> The plateau is hardware. The gap below 2× is overhead + serial reduce.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank You

### Next week: GPU Architecture & CuPy (Week 6)

*YZM 513 – İstanbul Medeniyet Üniversitesi – Fall 2026*
