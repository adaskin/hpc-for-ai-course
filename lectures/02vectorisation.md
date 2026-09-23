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
  footer {
    font-size: 16px;
    opacity: 0.5;
  }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# YZM 513 – High-Performance Programming for AI

### Week 2: Vectorisation & JIT – NumPy + Numba

**İstanbul Medeniyet Üniversitesi**
AI Engineering M.Sc. – Fall 2026

Ammar Daşkın

---

## Recap: Where We Left Off

Last week you profiled three deliberately slow functions:

| Function | Problem |
|----------|---------|
| `normalize_rows` | Double Python loop for L2 norm |
| `pairwise_distances` | Triple Python loop |
| `softmax` | Three sequential Python loops |

You identified **why** they're slow:
> Python interpreter dispatching each `+`, `*`, `/` individually.

---

**Today we fix them.** Three strategies, increasing power:

1. **NumPy vectorisation** → move the loop into C/BLAS
2. **Numba JIT** → compile your loop to machine code
3. **Numba parallel** → compile + spread across cores

---

## Today's Goal

Take each Week 1 function and produce **four versions**:

```text
Python loop  →  NumPy  →  Numba @njit  →  Numba parallel
   (slow)       (fast)     (faster?)       (fastest?)
```

Then benchmark all four and understand **when each tool is appropriate**.

> By the end of today you should be able to look at a slow Python
> function and know exactly which strategy to reach for.

---

<!-- _class: lead -->

# Part I
# NumPy Vectorisation

---

## Why loops over NumPy arrays Are Slow (Recap)

```python
for i in range(1_000_000):
    s += a[i] * b[i]
```

Each iteration through the interpreter:
1. Fetch bytecode for `BINARY_SUBSCR` → type-check `a`, check index
2. Fetch bytecode for `BINARY_MULTIPLY` → look up `__mul__`, dispatch
3. Fetch bytecode for `INPLACE_ADD` → look up `__iadd__`, allocate
4. Store result, update refcounts, advance loop counter

**~100+ bytecode instructions per iteration.**

NumPy does the same math in a **compiled C loop** with SIMD instructions.
One function call replaces 1 000 000 interpreter iterations.

---

## What exactly are we comparing?
```python
# A. Pure Python lists
xs = [1.0, 2.0, 3.0]
ys = [4.0, 5.0, 6.0]
for i in range(n):
    z[i] = xs[i] * ys[i]

# B. NumPy arrays + Python loop (Week 1 anti-pattern)
xs = np.array([1.0, 2.0, 3.0])
for i in range(n):
    z[i] = xs[i] * ys[i]   # xs[i] creates a NumPy scalar

# C. NumPy vectorised
z = xs * ys                # one compiled C loop
```


---
## NumPy Array Memory Model

A NumPy array is a **contiguous block of memory** + metadata:

```text
ndarray
├── data pointer  → [1.0][2.0][3.0][4.0][5.0]  ← contiguous float64
├── dtype         → float64 (8 bytes per element)
├── shape         → (5,)
├── strides       → (8,)  ← bytes to step to next element
└── flags         → C_CONTIGUOUS, ALIGNED, ...
```

| Attribute | Meaning |
|-----------|---------|
| `arr.shape` | Dimensions, e.g. `(1000, 50)` |
| `arr.dtype` | Element type: `float32`, `float64`, `int64`, ... |
| `arr.strides` | Bytes to jump per axis |
| `arr.flags['C_CONTIGUOUS']` | Row-major, contiguous in memory |

<!-- note:
Why this matters:
- Contiguous memory → CPU cache lines are used efficiently.
- BLAS routines (called by np.dot) require contiguous data.
- If you slice weirdly (arr[::2]), you get a non-contiguous VIEW.
  Operations on it may be slower because they can't use SIMD.
- In AI: PyTorch tensors have the same layout. .contiguous() exists
  for exactly this reason.
-->

---

## C-Contiguous vs. F-Contiguous

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
# C-contiguous (row-major): rows are contiguous
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
# Memory: [0,1,2,3,4,5,6,7,8,9,10,11]

b = np.asfortranarray(a) 
c = np.arange(12).reshape(3, 4,order='F')
# F-contiguous (column-major): columns are contiguous
# Memory: [0,4,8,1,5,9,2,6,10,3,7,11]
```

- **C-contiguous:** default in NumPy. Iterate over last axis fastest.
- **F-contiguous:** used by BLAS/LAPACK (Fortran heritage).
- **For this course:** C-contiguous is fine. Just know it exists.

<!-- note:
you will encounter this in PyTorch when you see
tensor.contiguous() or tensor.is_contiguous(). The key insight: memory layout affects cache performance.
-->

---

## The Vectorisation Principle

> **Replace element-wise Python loops with whole-array operations.**

```python
# ❌ Slow: Python loop
result = np.zeros(n)
for i in range(n):
    result[i] = a[i] ** 2 + 2 * a[i] + 1

# ✅ Fast: vectorised
result = a ** 2 + 2 * a + 1
```

- The `**`, `*`, `+` operators on arrays call compiled C code.
- No Python-level loop. No per-element type checking.
- CPU processes multiple elements per instruction (SIMD).

> This single habit eliminates 90% of Python performance problems in AI.

---

## Broadcasting Rules

NumPy can operate on arrays of **different shapes** without copying data:

```python
a = np.random.rand(1000, 50)   # (1000, 50)
b = np.random.rand(50)         # (50,)

c = a + b   # b is "stretched" to (1000, 50) – no copy made
```

---

**Rules (right to left):**
1. Align shapes from the right.
2. Dimensions match if they're equal **or one of them is 1**.
3. The size-1 dimension is "stretched" to match.

```text
  (1000, 50)
+     (50,)
───────────
  (1000, 50)  ✓

  (1000, 1)
+    (1, 50)
───────────
  (1000, 50)  ✓  (outer-product-like)
```

<!-- note:
Broadcasting is how you avoid explicit loops for row/column operations.
Example: normalising each row by its mean:
  X - X.mean(axis=1, keepdims=True)
The keepdims=True keeps the shape as (n, 1) so broadcasting works.
This is exactly what we need for normalize_rows.
-->

---

## Fixing `normalize_rows` with NumPy

**Before (Week 1):**
```python
def normalize_rows(matrix):
    result = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        norm = 0.0
        for j in range(matrix.shape[1]):
            norm += matrix[i, j] ** 2
        norm = norm ** 0.5
        for j in range(matrix.shape[1]):
            result[i, j] = matrix[i, j] / norm
    return result
```

---
**After:**
```python
def normalize_rows_numpy(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # (n, 1)
    return matrix / norms  # broadcasting: (n, d) / (n, 1)
```

**Two lines. No loops. ~200× faster.**

<!-- note:
why this works:
1. np.linalg.norm computes the L2 norm along axis=1 → shape (n,).
2. keepdims=True keeps it as (n, 1) for broadcasting.
3. matrix / norms broadcasts: each row divided by its scalar norm.
4. All done in compiled C. No Python loop.
-->

---

## Fixing `softmax` with NumPy

**Before:** Three Python loops (find max, compute exp, divide).

**After:**
```python
def softmax_numpy(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)  # stability
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)
```

Three vectorised operations. Each one is a compiled C loop over the array.

> The `keepdims=True` pattern appears again. It's the key to
> row-wise and column-wise operations without loops.

<!-- note:
the numerical stability trick: subtracting the max before exp.
This is the same trick used in PyTorch's nn.functional.softmax.
"Why do we subtract the max?" → prevents overflow in exp().
-->

---

## Fixing `pairwise_distances` with NumPy

This one is trickier. The triple loop computes $\|a_i - b_j\|$ for all $(i, j)$.

**Trick: expand the squared distance.**

$$\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2\, a \cdot b$$

```python
def pairwise_distances_numpy(A, B):
    # A: (n, d), B: (m, d)
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)  # (n, 1)
    B_sq = np.sum(B ** 2, axis=1, keepdims=True)  # (m, 1)
    cross = A @ B.T                                # (n, m)
    dist_sq = A_sq + B_sq.T - 2 * cross           # (n, m)
    return np.sqrt(np.maximum(dist_sq, 0))         # clamp negatives
```

> Replaced an $O(n \cdot m \cdot d)$ Python triple loop with
> **one matrix multiply** (calls BLAS).

<!-- note:
This is the key insight for this function. The algebraic expansion
turns a triple loop into a matmul. BLAS is highly optimised.
The np.maximum(dist_sq, 0) handles floating-point rounding that
could make dist_sq slightly negative.
This pattern appears in k-means, nearest-neighbour search, and
attention mechanisms. 
-->

---

## Benchmark: Python-numpy loop vs. NumPy

> Full script: `class-examples/week02/benchmark_numpy.py`

Typical results (2000×50 matrix, 500×50 matrix):

```
============================================================
Function                   Python (ms)   NumPy (ms)    Speedup
============================================================
normalize_rows                   52.12       0.4016       130×
softmax                          36.18       0.3853        94×
pairwise_distances             1946.91       2.7128       718×
============================================================
```


<!-- note:
"Is NumPy always enough? When would you still need something more?"
when the operation isn't expressible as a combination of
standard array operations. Custom recurrence, irregular access patterns,
conditional logic that depends on element values. That's where Numba comes in.
-->

---

## When NumPy Is Enough

✅ Element-wise operations: `a * b + c`
✅ Reductions: `np.sum`, `np.mean`, `np.max` along an axis
✅ Linear algebra: `A @ B`, `np.linalg.solve`, `np.dot`
✅ Broadcasting patterns: row/column normalisation, outer products
✅ Anything with a BLAS / LAPACK equivalent

> **Rule of thumb:** if you can express it as a combination of
> array operations, broadcasting, and `np.linalg`, use NumPy.
> Don't reach for anything else.

---

## When NumPy Is NOT Enough

❌ **Data-dependent branching:**
```python
for i in range(n):
    if arr[i] > threshold:
        arr[i] = some_function(arr[i])
    else:
        arr[i] = other_function(arr[i])
```
*(Hard to vectorise if the logic is complex.)*

---

❌ **Sequential recurrences:**
```python
for t in range(1, T):
    x[t] = alpha * x[t-1] + beta * input[t]
```
*(Each step depends on the previous. Can't parallelise.)*

❌ **Irregular access patterns:** graph traversals, sparse updates.

> For these → **Numba JIT** (next section).

---

<!-- _class: lead -->

# Part II
# Numba: JIT Compilation

---

## What Is Numba?

> **Numba** translates Python functions to optimised machine code at runtime.

```text
  Your Python function
       │
       ▼
  Numba analyses the code (type inference)
       │
       ▼
  Generates LLVM IR → machine code
       │
       ▼
  Compiled function (runs at C speed)
```

- **No C/C++ needed.** You write Python. Numba compiles it.
- First call is slow (compilation). Subsequent calls are fast.
- Works best with NumPy arrays, scalars, and simple control flow.

---

## Your First Numba Function

```python
from numba import njit
import numpy as np

@njit
def pairwise_distances_numba(A, B):
    n = A.shape[0]
    m = B.shape[0]
    d = A.shape[1]
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            total = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                total += diff * diff
            D[i, j] = total ** 0.5
    return D
```

> **Same code as the slow version.** Just added `@njit`.
> The loop is now compiled to machine code. ~100–500× faster than pure Python.

<!-- note:
Key point: we did NOT change the algorithm. We didn't vectorise.
We kept the triple loop. Numba compiles it to efficient machine code.
This is powerful when you CAN'T vectorise (complex logic, recurrences).
The @njit decorator = @jit(nopython=True). We'll explain nopython mode next.
-->

---

## `@jit` vs `@njit`

```python
from numba import jit, njit

@jit          # May fall back to "object mode" (slow) if it can't compile
def f1(x): ...

@njit         # Strict: compile or raise an error. Always fast.
def f2(x): ...
```

| Mode | Speed | Safety |
|------|-------|--------|
| `@jit` | Fast if compiled; slow fallback otherwise | Silent fallback |
| `@njit` (= `@jit(nopython=True)`) | Always fast | Raises error if it can't compile |

> **Always use `@njit`.** If Numba can't compile your function,
> you want to know immediately, not silently fall back to slow mode.

<!-- note:
In practice, @njit is what everyone uses. The "object mode" fallback
of @jit defeats the purpose.  "@njit is the default
choice. If it errors, you'll see what's unsupported and can refactor."
-->

---

## First Call vs. Subsequent Calls

```python
import time

A = np.random.rand(200, 50)
B = np.random.rand(500, 50)

# First call: includes compilation
t0 = time.perf_counter()
D = pairwise_distances_numba(A, B)
t1 = time.perf_counter()
print(f"First call (with compilation): {t1-t0:.3f} s")

# Second call: already compiled
t0 = time.perf_counter()
D = pairwise_distances_numba(A, B)
t1 = time.perf_counter()
print(f"Second call (compiled):        {t1-t0:.4f} s")
```

---

Typical output:
```text
First call (with compilation): 0.842 s
Second call (compiled):        0.0031 s
```

> Compilation is a one-time cost. In a training loop, you call the
> function thousands of times → amortised cost is negligible.

<!-- note:
Important  to understand:
- Don't benchmark the first call. It includes compilation.
- In production (training loops), the function is called millions of times.
- Numba also has @njit(cache=True) to cache compiled code to disk.
- For this course, just call twice: first to compile, second to benchmark.
-->

---

## What Numba CAN Compile

✅ NumPy arrays and scalars
✅ Arithmetic: `+`, `-`, `*`, `/`, `**`
✅ Comparisons: `<`, `>`, `==`
✅ `if` / `elif` / `else`, `for`, `while`
✅ Tuples, basic indexing: `a[i]`, `a[i, j]`
✅ `np.zeros`, `np.ones`, `np.arange`, `np.sqrt`, `np.exp`, ...
✅ Functions calling other `@njit` functions

---


## What Numba CANNOT Compile

❌ Python lists, dicts, sets (limited support)
❌ String operations
❌ Arbitrary Python objects / classes
❌ `print()` (limited support in newer versions)
❌ Dynamic typing: variables must have consistent types

<!-- note:
The rule of thumb: if your function is "numeric Python" (arrays, loops,
arithmetic, conditionals), Numba can compile it. If it's "Pythonic Python"
(dicts, strings, classes, duck typing), it can't.
For AI work, most hot loops are numeric → Numba works well.
-->

---

## Numba with Parallelism: `@njit(parallel=True)`

```python
from numba import njit, prange

@njit(parallel=True)
def pairwise_distances_parallel(A, B):
    n = A.shape[0]
    m = B.shape[0]
    d = A.shape[1]
    D = np.zeros((n, m))
    for i in prange(n):       # ← prange instead of range
        for j in range(m):
            total = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                total += diff * diff
            D[i, j] = total ** 0.5
    return D
```

---

- `prange` = **parallel range**. Numba distributes iterations across CPU cores.
- Only the **outer loop** is parallelised (the `i` loop).
- Requires `parallel=True` in the decorator.

> This is a **preview of Week 3** (CPU parallelism). For today,
> just know it exists and gives you multi-core speedup for free.

<!-- note:
Key point: prange tells Numba "these iterations are independent."
Each row i of the output depends only on A[i] and all of B. No data
dependency between iterations → safe to parallelise.
On an 8-core laptop, expect ~6-7x speedup over the serial @njit version.
This connects to Week 3 where we'll discuss threading/multiprocessing
more generally. For now, it's "one keyword to use all your cores."
-->

---

## Benchmark: All Four Versions

> Full script: `class-examples/week02/benchmark_all.py`

Typical results (`pairwise_distances`, A: 200×50, B: 500×50):

```
=================================================================
Version                           Time (ms)      Speedup
=================================================================
Python loop                         1991.91           1×
NumPy (matmul trick)                 1.1128        1790×
Numba @njit (serial)                 4.3996         453×
Numba parallel (prange)              1.0921        1824×
=================================================================
```

<!-- note:
Interesting observations:
1. NumPy with the matmul trick BEATS serial Numba here because BLAS
   is extremely optimised for matmul.
2. Numba parallel BEATS NumPy here because the problem is small enough
   that BLAS overhead doesn't help, but 8 cores do.
3. For larger problems, NumPy/BLAS may win again due to cache efficiency.
4. The "best" tool depends on the problem. Profile and compare.
This is the key lesson: there's no single "fastest" tool. Context matters.
-->

---

## Decision Flowchart

```text
Is it a standard array operation?
(matmul, reduction, element-wise, broadcasting)
    │
    ├── YES → Use NumPy / BLAS
    │
    └── NO (custom loop, recurrence, branching)
         │
         ├── Can you express it as array ops with a trick?
         │   (e.g., algebraic expansion, einsum)
         │       ├── YES → Use NumPy
         │       └── NO ↓
         │
         ├── Use Numba @njit
         │   (compile the loop as-is)
         │       │
         │       └── Iterations independent?
         │           ├── YES → @njit(parallel=True) + prange
         │           └── NO  → @njit (serial)
         │
         └── Still not fast enough? → GPU (Weeks 5–8)
```

---

## NumPy vs. Numba: Summary

| Criterion | NumPy | Numba |
|-----------|-------|-------|
| Best for | Standard ops, linear algebra | Custom loops, recurrences |
| Code change | Rewrite as array ops | Add `@njit` decorator |
| Compilation | None (pre-compiled C) | JIT (first call slow) |
| Parallelism | Implicit (BLAS threads) | Explicit (`prange`) |
| Readability | High (if you know the idiom) | High (looks like original) |
| Limitation | Must fit array-op pattern | Numeric types only |

> **They are complementary, not competing.**
> Use NumPy where you can. Use Numba where you must.

---

<!-- _class: lead -->

# Part III
# Advanced NumPy: Tips for AI

---

## `einsum`: Einstein Summation

For complex contractions, `np.einsum` is often the cleanest vectorisation:

```python
# Pairwise distances (alternative to the matmul trick)
D_sq = np.einsum('ij,ij->i', A, A)[:, None] \
     + np.einsum('ij,ij->i', B, B)[None, :] \
     - 2 * A @ B.T

# Attention scores: Q @ K^T / sqrt(d)
scores = np.einsum('bqd,bkd->bqk', Q, K) / np.sqrt(d)
```

- `'bqd,bkd->bqk'` = batched dot product over the `d` dimension.
- Extremely useful for attention, bilinear forms, contractions.
- PyTorch equivalent: `torch.einsum`.

<!-- note:
just know einsum exists and is
useful for attention-style operations you'll see in transformers.
The notation is compact but powerful. you'll encounter it in
PyTorch code and papers.
-->

---


> **`np.einsum` is NumPy’s implementation of Einstein summation notation.**

In Einstein notation, if an index appears twice on the right-hand side and does **not** appear in the output, it is **summed over**. The `einsum` string is just a compact way to write that index notation.

---

## General rule

For a call like:

```python
np.einsum('ij,jk->ik', A, B)
```

---

the mathematical formula is:

$$  
C_{ik} = \sum_{j} A_{ij} B_{jk}
$$

Mapping:

| `einsum` part | Math meaning |
|---|---|
| `ij` | indices of first input \(A_{ij}\) |
| `jk` | indices of second input \(B_{jk}\) |
| `->ik` | output indices \(C_{ik}\) |
| repeated `j` not in output | summation index: \(\sum_j\) |

---

So the rule is:

$$
\text{output} = \sum_{\text{indices not in output}} \prod_{\text{inputs}} \text{input}
$$

---

## Basic examples with formulas

### Matrix multiplication

```python
np.einsum('ij,jk->ik', A, B)
```

$$ 
C_{ik} = \sum_j A_{ij} B_{jk}
$$

---

### Dot product

```python
np.einsum('i,i->', x, y)
```

$$  
s = \sum_i x_i y_i
$$

---

### Outer product

```python
np.einsum('i,j->ij', x, y)
```

$$  
C_{ij} = x_i y_j
$$  

---

### Transpose

```python
np.einsum('ij->ji', A)
```

$$  
B_{ji} = A_{ij}
$$

---

### Trace

```python
np.einsum('ii->', A)
```

$$  
t = \sum_i A_{ii}
$$

---

## Attention example

For attention scores:

```python
scores = np.einsum('bqd,bkd->bqk', Q, K) / np.sqrt(d_model)
```

Mathematically:

$$  
S_{bqk} = \sum_d Q_{bqd} K_{bkd}
$$ 

Then the attention output:

```python
output = np.einsum('bqk,bkd->bqd', weights, V)
```

$$  
O_{bqd} = \sum_k W_{bqk} V_{bkd}
$$

This is exactly the batched matrix multiply at the heart of transformer attention.

---

## Pairwise distances with `einsum`

The squared Euclidean distance is:

$$  
D_{ij}^2 = \sum_k (A_{ik} - B_{jk})^2
$$

Expand it:

$$
D_{ij}^2 =
\sum_k A_{ik}^2
+
\sum_k B_{jk}^2
-
2 \sum_k A_{ik} B_{jk}
$$

In `einsum`:

```python
A_sq = np.einsum('ij,ij->i', A, A)   # sum_j A_{ij}^2
B_sq = np.einsum('ij,ij->i', B, B)   # sum_j B_{ij}^2
cross = np.einsum('ik,jk->ij', A, B) # sum_k A_{ik} B_{jk}

dist_sq = A_sq[:, None] + B_sq[None, :] - 2 * cross
D = np.sqrt(np.maximum(dist_sq, 0))
```

---

So:

$$
A\_sq_i = \sum_j A_{ij}^2
$$

$$
B\_sq_j = \sum_k B_{jk}^2
$$

$$
cross_{ij} = \sum_k A_{ik} B_{jk}
$$

$$
D_{ij}^2 = A\_sq_i + B\_sq_j - 2\, cross_{ij}
$$

---

## Summary


$$
\texttt{np.einsum('ij,jk->ik', A, B)}
\quad\equiv\quad
C_{ik} = \sum_j A_{ij} B_{jk}
$$

$$
\texttt{np.einsum('bqd,bkd->bqk', Q, K)}
\quad\equiv\quad
S_{bqk} = \sum_d Q_{bqd} K_{bkd}
$$

$$
\texttt{np.einsum('bqk,bkd->bqd', W, V)}
\quad\equiv\quad
O_{bqd} = \sum_k W_{bqk} V_{bkd}
$$


> **Key idea:** repeated index not in output = summation.  
> `einsum` is just index notation with letters instead of subscripts.

---

## Avoiding Copies: Views vs. Copies

```python
a = np.arange(100)

b = a[10:20]       # VIEW: no copy, shares memory with a
c = a[10:20].copy() # COPY: independent array

b[0] = 999          # Modifies a[10] too!
```

- **Views** are free (no memory allocation). Prefer them.
- **Copies** allocate new memory. Use only when you need independence.
- Transposing: `a.T` is a view (just changes strides). Free.
- Reshaping: `a.reshape(...)` is a view if possible.

> In AI: PyTorch's `.view()` vs `.reshape()` vs `.contiguous()`
> follow the same logic.

<!-- note:
Brief mention. The key insight: not all operations allocate memory.
Views are free. This matters when you're processing large tensors
and want to avoid OOM. Connect to PyTorch's .view() and .reshape().
-->

---

## In-Place Operations

```python
# Allocates a new array each time:
result = a + b
result = result * 2
result = result - c

# In-place: reuses the same memory
result = a + b
result *= 2
result -= c
```

- `*=`, `-=`, `+=` modify the array in place. No new allocation.
- Matters when arrays are large (GB-scale tensors).
- PyTorch equivalent: `tensor.add_(value)` (trailing underscore = in-place).

---

## dtype Matters

```python
a64 = np.random.rand(1000, 1000)          # float64: 8 MB
a32 = np.random.rand(1000, 1000).astype(np.float32)  # float32: 4 MB
```

- `float32` uses **half the memory** and is **faster** on GPUs.
- Most deep learning uses `float32` by default (or `float16`/`bfloat16`).
- NumPy defaults to `float64`. PyTorch defaults to `float32`.
- Be explicit: `np.zeros(n, dtype=np.float32)`.

> This connects to **Week 12: Mixed Precision Training**.

---

<!-- _class: lead -->

# Part IV
# Hands-On Exercise

---

## Exercise: "Make This 100× Faster"

📁 **File:** `class-examples/week02/exercise_week2.py`

You are given three functions from Week 1 (same ones you profiled).


### Your task:

1. **Vectorise** each function using NumPy.
   - `normalize_rows` → use `np.linalg.norm` + broadcasting
   - `softmax` → use `np.exp` + broadcasting
   - `pairwise_distances` → use the matmul trick

---

2. **JIT-compile** `pairwise_distances` with Numba:
   - Add `@njit` → benchmark
   - Add `@njit(parallel=True)` + `prange` → benchmark

---

3. **Benchmark all versions** using `timeit` or `time.perf_counter`.
   Fill in this table:

| Function | Python | NumPy | Numba | Numba parallel |
|----------|-------:|------:|------:|---------------:|
| normalize_rows | ? | ? | ? | ? |
| softmax | ? | ? | ? | ? |
| pairwise_distances | ? | ? | ? | ? |

---

## Exercise: Bonus Challenges

4. **Numba for softmax?** Try `@njit` on the original softmax.
   - Does it work? Why or why not?
   - *(Hint: `np.exp` is supported. But what about the `max` loop?)*

---

5. **`einsum` for attention:**
   Given `Q, K, V` of shape `(batch, seq_len, d_model)`:
   ```python
   scores = np.einsum('bqd,bkd->bqk', Q, K) / np.sqrt(d_model)
   weights = softmax_numpy(scores)
   output = np.einsum('bqk,bkd->bqd', weights, V)
   ```
   Time it vs. a Python-loop implementation.

---

6. **Cache:** Add `@njit(cache=True)` to your Numba functions.
   Run the script twice. Is the second run faster to start?

<!-- note:
Bonus 4 is a trick question. Numba CAN compile the softmax loop
(all operations are supported). But NumPy is still faster here because
np.exp is already optimised with SIMD. The lesson: Numba isn't always
the answer. Sometimes the "dumb" vectorised version wins.
Bonus 5 previews attention from transformers. Those who haven't seen
attention will see it in their
ML courses.
-->

---

## Exercise Debrief

**Key observations to discuss:**

1. **NumPy wins for standard patterns.** `normalize_rows` and `softmax`
   are textbook broadcasting. NumPy is unbeatable.

2. **Numba wins for custom loops** that can't be expressed as array ops.
   If you didn't know the matmul trick, `@njit` on the triple loop
   gets you 1000× with zero algorithmic insight.

3. **`parallel=True` helps when iterations are independent** and the
   problem is large enough to amortise thread overhead.

4. **There is no universal "fastest."** Profile, try, compare.

> The skill is not "know Numba" or "know NumPy."
> The skill is **knowing which to reach for and when.**

---

## Key Takeaways

1. **Vectorise first.** Replace Python loops with array operations.
   Broadcasting + `keepdims=True` handles 90% of cases.

2. **Numba is your escape hatch** when vectorisation isn't possible.
   `@njit` compiles your loop to machine code with zero algorithm change.

3. **`prange` gives free multi-core parallelism** for independent iterations.

4. **Benchmark everything.** The "obvious" fastest solution isn't always fastest.

5. **Next week:** we go deeper on parallelism — `threading`,
   `multiprocessing`, the GIL, and when threads vs. processes matter.

---

## Before Next Week

- ✅ Install (if you haven't): `pip install multiprocessing-utils` *(optional)*
- 📖 Read: Python `concurrent.futures` docs (10 min skim)
  → https://docs.python.org/3/library/concurrent.futures.html
- 📖 Optional: PEP 703 summary (free-threaded CPython)
  → https://peps.python.org/pep-0703/
- 🤔 Think about: where in your own code / thesis work do you have
  loops that could be parallelised?

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank You

### Next week: Threading, Multiprocessing & the GIL

*YZM 513 – İstanbul Medeniyet Üniversitesi – Fall 2026*

---

## Class-examples files

### `class-examples/week02/benchmark_numpy.py`
```python

```

### `class-examples/week02/benchmark_all.py`

```python

```

### `class-examples/week02/exercise_week2.py`

```python
```

