# YZM 513 -- Project 1: High-Performance Similarity Search Engine

## Weeks 1-5 | Due: Week 7 (before midterm)

**Format:** Groups of 2  
**Deliverables:** Report (10-15 pages, PDF) &middot; In-class presentation (15 min + 5 min Q&A) &middot; Code repository (GitHub)  
**Weight:** 30% of course grade (20% report + 10% presentation)

---

## 0. The Problem

You have a corpus of **N documents** (LLM outputs, paper abstracts, embeddings--your choice). A user submits a query. Your system must return the **top-k most similar documents** fast enough to be interactive.

Your naive Python implementation takes seconds. Your goal: **make it fast enough to be usable**, and **tell the story of how you got there**.

This is not a list of exercises. This is a system you are building. Every technique you apply should be motivated by a bottleneck you identified, and every design choice should be justified with data.

---

## 1. AI Usage Policy

AI assistants (ChatGPT, Copilot, Claude, Kimi, etc.) are **encouraged** for writing code, generating plots, and drafting text. However:

- Every figure must be explained in your report and presentation. You must understand what the plot shows and why it looks that way.
- Every code block must be explained line-by-line during the presentation.
- You must be able to answer questions about any part of your code or results during Q&A.
- **Penalty:** -15 points if you cannot explain a figure or code block you submitted.

---

## 2. Constraints

- Your system must run on a **laptop or free Google Colab** (no paid GPU).
- Your corpus must have **>= 5,000 documents**.
- Your search must return **correct results** (verified against baseline).
- Your report must be **reproducible**: include `requirements.txt` and a README with run instructions.
- Your presentation demo must **run live** (no pre-recorded videos).
- All benchmarks must use **warmup runs** and **>= 5 timed iterations**.

---

## 3. Dataset (Choose One)

Use a dataset from **2025-2026** or a realistic synthetic dataset representing a modern AI workload.

| Option | Description |
|--------|-------------|
| **A. LLM Output Corpus** *(recommended)* | Generate 5,000-20,000 text responses from a modern LLM (GPT-4, Claude, Llama, Qwen) on diverse prompts. Each response is a "document." Goal: find semantically similar LLM outputs. |
| **B. Academic Abstracts (2025-26)** | Scrape recent abstracts from arXiv (cs.AI, cs.CL, cs.LG), OpenReview (ICLR/NeurIPS/ICML 2025/26), or ACL Anthology. |
| **C. KV-Store / RAG Embeddings** | Vector database query logs, prompt-response pairs, or embedding cache dumps from a RAG system. |
| **D. Synthetic (with justification)** | Generate embeddings from a known distribution. Must justify in report why this represents a real workload. |

**Minimum requirements:**

| Criterion | Minimum |
|-----------|---------|
| Documents / items | 5,000 |
| Average text length | 50 tokens |
| Embedding dimension (if pre-computed) | 128-768 |
| Queries for search | 50-200 |

**Include in report:** source, statistics, sample items, preprocessing steps.

---

## 4. Project Phases

### Phase 1: Baseline & Profiling

*Concepts used: Weeks 1 (profiling, Python performance limits)*

1. Convert your dataset into vector representations (sparse TF-IDF and/or dense embeddings).
2. Implement a **naive search** in pure Python (nested loops, element-wise cosine similarity).
3. **Profile it** with `cProfile` and `line_profiler`. Identify the top-3 bottlenecks.
4. Record the baseline time.

> **Deliverable:** Baseline timing + profiling screenshot + identified bottlenecks.
> **Key question:** Where is the time spent? Is it the similarity computation, the sorting, the data access pattern?

---

### Phase 2: High-Performance Matrix Operations & Dimensionality Reduction

*Concepts used: Weeks 1-2 (vectorisation, Numba JIT, BLAS, memory layout)*

This phase has two parts. Both are required.

#### 2A. Sparse vs. Dense: Representation Matters

Convert your corpus to both sparse (TF-IDF, CSR format) and dense (embeddings) representations. Compare:

- Memory footprint at different corpus sizes (1K, 5K, 10K, 20K docs)
- Dot product / matrix multiply time for sparse-sparse, sparse-dense, dense-dense
- When is each representation the right choice?

> You must implement the dot product for each combination yourself (not just call `sklearn`). Show the code.

#### 2B. Dimensionality Reduction: Choose and Justify

Your embedding matrix may be large (N x 768). You need to reduce it for faster search. Compare **at least two** of the following:

| Method | What to implement |
|--------|-------------------|
| **Your own power-iteration SVD** | Implement the iterative algorithm yourself (top-k singular vectors). This is the "learn the algorithm" exercise. |
| **NumPy `np.linalg.svd`** | Full SVD. Use as ground truth. |
| **Scikit-learn `TruncatedSVD`** | Randomised solver, top-k only. |
| **Scikit-learn `PCA`** | Full or incremental. |

For each method you implement, report:
- Time vs. number of components *k*
- Time vs. matrix size
- Reconstruction error (Frobenius norm) at *k* = 50, 100, 200
- Memory usage

> **Design decision to justify:** Which method do you use for your final search system, and why?
> Discuss: Why is Truncated SVD faster than full SVD? When does PCA differ from SVD? What is the practical maximum *k* where Truncated SVD still wins?

#### 2C. Vectorisation vs. Element-Wise vs. JIT

Implement the **same matrix operation** (e.g., gram matrix X^T X, or cosine similarity matrix) in four ways:

1. Pure Python loops (element-wise)
2. NumPy vectorised (`X.T @ X`)
3. Numba `@njit` on the loop version
4. Numba `@njit(parallel=True)` with `prange`

Benchmark all four. Report speedup relative to the Python baseline.

> **Key question:** When does Numba beat NumPy, and when does it lose? Why?

---

### Phase 3: Parallelism & Concurrency

*Concepts used: Weeks 3-5 (threads, processes, GIL, async, Dask/Ray)*

Starting from your Phase 2 code, apply **at least 3** parallelism strategies from the table below. You must justify *why* you chose those three and *why not* the others.

| Strategy | When it's the right choice |
|----------|---------------------------|
| NumPy vectorisation (BLAS) | Always start here. Implicit parallelism via BLAS threads. |
| Numba `parallel=True` + `prange` | Custom loops that can't be vectorised |
| `ThreadPoolExecutor` | I/O-bound tasks (API calls, file reads) or NumPy/BLAS ops (GIL released) |
| `ProcessPoolExecutor` | CPU-bound pure Python (GIL blocks threads) |
| `asyncio` | Massive concurrent I/O (>100 tasks, e.g., batch API queries) |
| Dask | Data larger than RAM, out-of-core computation |
| Ray (`@ray.remote` / Actors) | Distributing many independent tasks, stateful workers |

For each strategy you implement:
- Show the code
- Benchmark against baseline and against your best Phase 2 version
- Explain **why it helped** (or didn't)
- Connect to course concepts: GIL, Amdahl's Law, memory bandwidth, BLAS

> **Design decision to justify:** For your specific workload, which parallelism strategy gives the best speedup per unit of code complexity? Would your answer change if the corpus were 10x larger?

**Bonus (+3 pts):** Implement more than 3 strategies and compare all of them.

---

### Phase 4: The Final System & Scaling

Bring it all together:

1. Your final search pipeline: data -> embedding -> dimensionality reduction -> parallel search -> top-k results.
2. **Scaling test:** Measure search time at corpus sizes 1K, 5K, 10K, 20K. Plot the scaling curve.
3. **Correctness check:** Verify top-k results match the baseline (tolerance 1e-5).
4. **Live demo:** Your fastest version, running in real time.

> **Key question:** Does your system scale linearly? If not, where does it break down? What would you do with 10x more data?

---

## 5. Required Figures (5-7 total)

You don't need 15 plots. You need **5-7 figures that tell a story**. Every figure must have a caption that states a **conclusion**, not just a description.

| # | Figure | What it shows |
|---|--------|---------------|
| 1 | Profiling output (flame graph or `cProfile` top functions) | Where the bottlenecks are in the baseline |
| 2 | Sparse vs. dense: memory + time comparison | The representation trade-off |
| 3 | Dimensionality reduction: time vs. *k* (your methods) | Which method you chose and why |
| 4 | Vectorisation speedup: Python vs. NumPy vs. Numba vs. Numba-parallel | The "1000x speedup" story |
| 5 | Parallel scaling: speedup vs. number of workers | Does it match Amdahl's Law? |
| 6 | Final search: time vs. corpus size (scaling curve) | How your system scales |
| 7 | *(optional)* Benchmark bar chart: all versions compared | The "money shot" |

> X "Figure 3: Time vs. corpus size"
> OK "Figure 3: Our vectorised search scales linearly to 20K docs (0.8 ms per 1K docs), while the baseline scales quadratically and becomes unusable above 5K docs."

---

## 6. Report Structure

| Section | Pages | Content |
|---------|:-----:|---------|
| **1. Introduction** | 1 | Dataset, problem statement, hardware/software environment |
| **2. Baseline & Profiling** | 1-2 | Naive implementation, profiling results, identified bottlenecks |
| **3. Matrix Operations & Dimensionality Reduction** | 3-4 | Sparse vs. dense, SVD/PCA comparison, vectorisation vs. JIT |
| **4. Parallelism & Concurrency** | 3-4 | Your chosen strategies, benchmarks, GIL/Amdahl analysis |
| **5. Design Decisions** | 1-2 | What you chose, what alternatives you considered, why (with data) |
| **6. Final System & Scaling** | 1-2 | End-to-end pipeline, scaling curve, correctness |
| **7. Discussion & Lessons** | 1 | What surprised you? What would you do differently? Connection to real AI systems. |
| **Appendix** | -- | Full code listing or GitHub link, `requirements.txt`, raw benchmark data |

---

## 7. Presentation Requirements

**Format:** 15 minutes + 5 minutes Q&A

Your presentation must tell a **performance story**:

1. **The problem:** "We needed to search N documents in under X ms."
2. **The baseline:** "Our naive implementation took Y seconds. Here's the profile."
3. **The journey:** "We tried A. It helped by Bx. Then we tried C..."
4. **The result:** "Our final system does it in Z ms. Here's the demo."
5. **The lessons:** "We learned that [GIL / memory bandwidth / Amdahl's Law] was the real bottleneck."

**Rules:**
- Every figure you show must advance the story.
- Every code snippet must explain *why* you wrote it that way.
- Include a **30-second live demo** of your fastest version.
- End with one slide: "What surprised us" and "What we'd do with more time."

---

## 8. Grading

### Report (20 points)

| Criterion | Weight | Excellent |
|-----------|:------:|-----------|
| **Working system & code** | 30% | Baseline + >=3 optimisations, all correct, reproducible |
| **Performance story** | 25% | Clear profiling -> bottleneck -> fix narrative. Figures tell the story. |
| **Design justification** | 20% | Every choice justified with data. Alternatives discussed. |
| **Technical depth** | 15% | Connects results to course theory (GIL, Amdahl, BLAS, memory bandwidth). |
| **Report quality** | 10% | Well-structured, reproducible, honest about limitations. |

### Presentation (10 points)

| Criterion | Weight | Excellent |
|-----------|:------:|-----------|
| Explanation of figures | 30% | Every figure clearly explained with insight |
| Code explanation | 25% | Can explain any code line-by-line |
| Technical depth & Q&A | 25% | Handles questions confidently, demonstrates understanding |
| Demo & storytelling | 20% | Smooth demo, clear narrative arc |

### Bonus (+3 points)

- Implement >3 parallelism strategies and compare all
- Include a roofline model analysis for your matrix operations
- Test on two different machines (laptop + Colab) and compare
- Implement approximate nearest neighbours (LSH, HNSW) as an additional comparison
- Implement both Dask and Ray and compare

### Penalties

| Penalty | Reason |
|---------|--------|
| -15 pts | Cannot explain a figure or code block during presentation |
| -10 pts | Missing dataset description or preprocessing steps |
| -10 pts | Unfair benchmarks (different hardware, no warmup, single run) |
| -5 pts | No `requirements.txt` or missing code |

---

## 9. Starter Code

See `starter_code/` directory in the course repository:

```
starter_code/
├── phase1_baseline/
│   ├── naive_search.py          # Pure Python cosine similarity search
│   └── profiling_template.py    # cProfile + line_profiler boilerplate
├── phase2_matrix_ops/
│   ├── sparse_dense_compare.py  # Skeleton for sparse vs dense
│   ├── svd_power_iteration.py   # Skeleton for your own SVD
│   └── vectorise_vs_loop.py     # Skeleton for 4-way comparison
├── phase3_parallelism/
│   ├── thread_pool_search.py    # ThreadPoolExecutor template
│   ├── process_pool_search.py   # ProcessPoolExecutor template
│   ├── async_search.py          # asyncio template
│   └── ray_search.py            # Ray template
├── benchmark_utils.py           # Timing, memory, plotting helpers
└── requirements.txt
```

---

## 10. Practical Tips

**Dataset:**
- Start with 5,000 documents for development; scale to 20,000 for final benchmarks.
- If using LLM outputs, cache them locally to avoid repeated API costs.
- `sklearn.datasets.fetch_20newsgroups` is acceptable as a fallback, but justify why.

**Benchmarking:**
```python
# Always warmup (critical for Numba JIT)
for _ in range(3):
    result = my_function(...)

# Multiple runs for reliability
import time
times = []
for _ in range(5):
    t0 = time.perf_counter()
    result = my_function(...)
    times.append(time.perf_counter() - t0)
print(f"Median: {sorted(times)[len(times)//2]:.4f} s")
```

**Colab:**
- Free CPU runtime is sufficient (no GPU needed).
- Colab has 2 cores -- adjust `max_workers` accordingly.
- Save results to Google Drive for persistence.
- Put `!pip install` in the first cell.

**Plotting:**
```python
import matplotlib.pyplot as plt
plt.xlabel("Number of Documents")
plt.ylabel("Execution Time (seconds)")
plt.title("Search Time vs. Corpus Size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("scaling.png", dpi=150, bbox_inches="tight")
```

---

## 11. FAQ

**Q: Can we use a pre-trained embedding model (e.g., sentence-transformers)?**
A: Yes, but embedding generation time does NOT count toward your benchmarks. Only the matrix operations and search phases are benchmarked.

**Q: Do we need to implement SVD from scratch?**
A: You must implement at least the power iteration method yourself. You can use `np.linalg.svd` for comparison, but your own implementation must be present and benchmarked.

**Q: Do we need to implement ALL parallelism strategies?**
A: No. Choose at least 3 and justify your choices. Implementing more earns bonus points.

**Q: Can we use AI to write our code?**
A: Yes, absolutely. But you must understand every line. During the presentation, you will be asked to explain any code you show.

**Q: What if our laptop is too slow?**
A: Use Google Colab (free). The starter code works there.

**Q: Can we work in groups of 3?**
A: No. Groups of 2 only. If you cannot find a partner, contact the instructor.

**Q: What counts as a "design decision"?**
A: Any point where you had multiple options and chose one. Examples: sparse vs. dense representation, Truncated SVD vs. PCA, threads vs. processes for your workload, chunk size for Dask, number of workers. For each: state what you chose, what you considered, and show data supporting your choice.

---

## 12. Timeline

| Week | Milestone |
|------|-----------|
| Week 5 | Form groups, choose dataset, complete Phase 1 (baseline + profiling) |
| Week 6 | Complete Phases 2-3 (matrix ops, dimensionality reduction, parallelism) |
| Week 7 | Complete Phase 4, submit report, present in class |

---

## 13. Submission Checklist

**Report:**
- [ ] PDF, 10-15 pages
- [ ] Dataset description with samples and statistics
- [ ] Baseline profiling evidence
- [ ] Sparse vs. dense comparison with figures
- [ ] Dimensionality reduction comparison (your SVD + at least one library method)
- [ ] Vectorisation vs. JIT comparison
- [ ] >=3 parallelism strategies with benchmarks
- [ ] Design decisions section
- [ ] Scaling curve
- [ ] Discussion connecting results to course concepts

**Code:**
- [ ] GitHub repository link
- [ ] `requirements.txt`
- [ ] `README.md` with instructions to reproduce
- [ ] All figures generated by code (not hand-drawn)
- [ ] Raw benchmark data (CSV or JSON)

**Presentation:**
- [ ] 15-minute slides
- [ ] Live demo prepared and tested
- [ ] Every figure rehearsed and explained
- [ ] Every code block rehearsed and explained

---

*Good luck. May your speedups be large, your bottlenecks be few, and your plots tell a story.*

**YZM 513 -- Istanbul Medeniyet Universitesi -- Fall 2026**
