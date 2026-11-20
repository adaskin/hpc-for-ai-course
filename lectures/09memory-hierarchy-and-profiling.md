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

# YZM 513 – High-Performance Programming for AI

### Week 9: The CUDA Memory Hierarchy & Profiling 🏛️
### Coalescing, shared memory, tiling, reductions

**İstanbul Medeniyet Üniversitesi**
AI Engineering M.Sc. – Fall 2026

Ammar Daşkın

---

## 🔁 Review: the kernel function

<div class="cols">
<div class="card blue">

### 🧩 What it is
A GPU function **meant to be called from CPU code**.

- cannot explicitly **return** values
- results are written into an **array passed to the function**

</div>
<div class="card orange">

### 🚀 How you launch it
The **thread hierarchy is declared at the call**:

- the number of **blocks**
- the number of **threads per block**
- the same kernel can be launched many times with different sizes

</div>
</div>

📖 [Numba CUDA: Writing Kernels](https://nvidia.github.io/numba-cuda/user/kernels.html)

---

## 🧭 Thread positioning: every thread runs the same code

When a kernel runs, **every thread executes the kernel function's code once**.

```C
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength){
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

     if(workIndex < vectorLength) {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}

int main(){
    ...
    // A, B, and C are vectors of 1024 elements
    // vectorLength is an integer storing number of elements in the vector
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    ...
}
```

---

## 🐍 Launching a kernel from Python

```python
from numba import cuda

@cuda.jit
def my_kernel(input, output):
    ...

## launch the kernel
my_kernel[num_thread_blocks, threads_per_block](in_array, out_array)
```

### Multi-dimensional grids and thread blocks

```python
@cuda.jit
def function(input, output):
    ...

## launch the kernel
function[(gridX, gridY), (blockX, blockY)](in_array, out_array)
```

---

## 🧮 Manual thread positioning

```python
# manual positioning
@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1
```

---

## 🎯 `cuda.grid`: let Numba do the arithmetic

- `numba.cuda.grid(ndim)` returns absolute position of the current thread in the entire grid blocks
- `numba.cuda.gridsize(ndim)` returns the absolute size (or shape) in threads of the entire grid of blocks
  
```python
#1D array
@cuda.jit
def increment_by_one(an_array):
    pos = cuda.grid(1)
    if pos < an_array.size:
        an_array[pos] += 1
#2D array
@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1
```

---

## 🔁 Review: memory management

**In C**, unified memory:

```C
 // Use unified memory to allocate buffers in C
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));
```

**In Numba**, device arrays:

```python
#an empty device array
numba.cuda.device_array(shape,dtype=np.float64,strides=None,order='C',stream=0,)
#from an array
numba.cuda.device_array_like(ary, stream=0)
# from numpy array
numba.cuda.to_device(obj, stream=0, copy=True, to=None)
```

[https://nvidia.github.io/numba-cuda/user/memory.html](https://nvidia.github.io/numba-cuda/user/memory.html)

---

## 🔄 Copying to and from the device

```python
ary = np.arange(10)
d_ary = cuda.to_device(ary)
# to a stream
stream = cuda.stream()
d_ary = cuda.to_device(ary, stream=stream)

#device to host
hary = d_ary.copy_to_host()
#host to device
ary = np.empty(shape=d_ary.shape, dtype=d_ary.dtype)
d_ary.copy_to_host(ary)
```

---

<!-- _footer: "Schedule note: original syllabus Week 8 topic, taught in Week 9." -->

## 🗺️ Where we are

```text
Week 1–2   one core, made fast        vectorise, Numba, BLAS
Week 3–4   one machine, many cores    threads, processes, async, pools
Week 5     many machines              Dask, Ray, cost of distribution
Week 6     one GPU, thousands of cores  architecture + CuPy
Week 7     midterm                    covers Weeks 1–6
Week 8     writing kernels            indexing, bounds, launches, transfers
Week 9     making kernels FAST        ← TODAY
Week 10    feeding the GPU            DataLoader pipelines
Week 11–12 many GPUs                  DDP, model parallelism
Week 13    deployment                 torch.compile
```

---

## 🎯 The week in one sentence

<div class="card blue big">

A kernel is fast when **each byte fetched from VRAM is used many times.**

</div>

🧱 **Tiling** is how you arrange that.
🔬 **Profiling** is how you prove it worked.

---

## 🧭 Today's arc

Everything starts from the **memory hierarchy**.

<div class="cols">
<div class="card green">

### 🔬 In detail
- **Coalescing**
- **Shared memory + tiling**

</div>
<div class="card gray">

### 👀 Briefly (know what it is)
- Reductions (softmax redeemed)
- Occupancy
- Profiling
- Streams: overlap copies and compute
- Lab + wrap-up

</div>
</div>

---

<!-- _class: lead -->

# 🔧 Hardware review

### How the GPU actually executes your grid

---

## 🧱 From Python grid to physical silicon

When you call `kernel[blocks, tpb](...)`, the hardware maps your software hierarchy onto physical units:

| Software (Your Code) | Hardware (The GPU) | Key Rule |
|---|---|---|
| **Grid** | Entire GPU | Blocks run in any order; no grid-wide sync |
| **Thread Block** | **Streaming Multiprocessor (SM)** | Assigned to **one SM** until finished |
| **Warp** (32 threads) | Warp Scheduler + Execution Cores | Executes in lockstep (**SIMT**) |
| **Thread** | CUDA Core / FP32 ALU | Has private registers |

*(A T4 GPU has **40 SMs**; an A100 has **108 SMs**.)*

---

## 🧵 The warp: the true unit of execution

- The SM does **not** schedule individual threads; it schedules **warps** (groups of **32 consecutive threads**).
- **Memory consequences:** Whenever a warp executes a load instruction, all 32 lanes issue their memory requests **together**.
  - *Global memory:* 32 requests are merged into cache lines (**coalescing**).
  - *Shared memory:* 32 requests hit 32 memory modules simultaneously (**banks**).
- **Latency hiding:** When Warp A stalls waiting on slow VRAM (~500 cycles), the SM switches to Warp B in **0 cycles**—if enough warps are resident (**occupancy**, later today).

---

## 💡 Why hardware matters today

<div class="cols3">
<div class="card blue">

### 🚰 Compute is cheap, memory is far
An SM has thousands of ALUs hungry for data, connected to VRAM by a narrow straw.

</div>
<div class="card orange">

### 📍 Blocks stay put
A block lives on **one SM**, so its threads can share the SM's fast on-chip SRAM.

</div>
<div class="card green">

### 🔢 32 is the magic number
Warp size = 32 lanes = 32 shared-memory banks = 128-byte cache line (32 floats).

</div>
</div>

Keep these three numbers in mind as we start counting bytes.

---

## 🗂️ GPU device memory types and locations
![memory types](img/w09-memory-types-scope.png)
[https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

---

## 🌍 Global memory

```python
@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]
```
A, B, and C are in global memory and are being accessed by this vector add kernel.

---

## 🤝 Shared memory
```python
@cuda.jit
def example_syncthreads(input_data, output_data):
    shared_data = cuda.shared.array(shape=128, dtype=np.int32)
    
    shared_data[cuda.threadIdx.x] = input_data[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x]
    cuda.syncthreads()

    if cuda.threadIdx.x == 0:
        sum = 0.0
        for x in shared_data: 
            sum = sum + x
        output_data[cuda.blockIdx.x] = sum
```
`cuda.syncthreads()` blocks all threads in the thread block until all threads have reached the call to `cuda.syncthreads()`

---

## 📐 Shared memory: static and dynamic allocation

### Static allocation
```python
from numba import cuda
import numpy as np

shared_array = cuda.shared.array(shape=1024, dtype=np.float32)
```
### Dynamic allocation

```python
from numba import cuda
import numpy as np

## setting shape=0 indicates that this array will be dynamically sized by the kernel launch's execution configuration
shared_array = cuda.shared.array(shape=0, dtype=np.float32)
```

---

## ⚡ Registers
- Registers are located on the SM and have thread local scope.
- Register usage is managed by the compiler
- registers are used for thread local storage during the execution of a kernel.

---

## 🏷️ Local memory
Local memory is thread local storage similar to registers and managed by NVCC, but the physical location of local memory is in the global memory space. 
- The ‘local’ label refers to its logical scope, not its physical location.

Local memory is used for thread local storage during the execution of a kernel. Automatic variables that the compiler is likely to place in local memory are:

- Arrays for which it cannot determine that they are indexed with constant quantities,

- Large structures or arrays that would consume too much register space,

- Any variable if the kernel uses more registers than available, that is register spilling.

---

## 🐌 Local memory is as slow as global memory

> Because the local memory space resides in device memory, local memory accesses have the same latency and bandwidth as global memory accesses and are subject to the same requirements for memory coalescing

---

## 🧊 Constant memory
- Constant memory has grid scope and is accessible for the lifetime of the application. 
- The constant memory resides on the device and is read-only to the kernel.
```python
const_array = numba.cuda.const.array_like(ary) # within the kernel code 
#creates a constant memory array 
# containing the data stored in the host array ary
```
- Resides in constant memory space,
- Has a distinct object per device,
- Is accessible from all the threads within the grid

---

## 🧊 Constant memory in code

```python
from numba import cuda
import numpy as np

host_array = np.zeros(128, dtype=np.float32)
## fill host_array with other data

@cuda.jit
def kernel(out):
    # created from the host array when the kernel is compiled
    const_array = cuda.const.array_like(host_array)

    # this access now goes through constant memory
    out[cuda.threadIdx.x] = const_array[cuda.threadIdx.x]
```
> Constant memory is useful for small amounts of data that each thread will use in a read-only fashion.

---

## 🗃️ Caches
- GPU devices have a multi-level cache structure which includes L2 and L1 caches.
- L1 cache is physically located on each SM and is the same physical space used by shared memory. If no shared memory is utilized by a kernel, the entire physical space will be utilized by the L1 cache.
- The L2 and L1 caches can be controlled via functions that allow the developer to specify various caching behaviors

---

<style scoped>section{font-size:24px} pre{font-size:15px;line-height:1.15}</style>

## 🏭 Inside a Streaming Multiprocessor (SM)

Every SM is an independent mini-processor with its own compute units and **on-chip memory**:

```text
┌───────────────────────────────────────────────────────┐
│ Streaming Multiprocessor (SM)                         │
│                                                       │
│  [ Warp Schedulers ]   [ CUDA Cores (FP32 / INT32) ]  │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Register File (65,536 × 32-bit regs on T4)      │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌────────────────────────┬────────────────────────┐  │
│  │ Shared Memory (SRAM)   │ L1 Data Cache (SRAM)   │  │
│  │ (User-managed)         │ (Hardware-managed)     │  │
│  └────────────────────────┴────────────────────────┘  │
└───────────────────────────▲───────────────────────────┘
                            │  Slow (~400–800 cycles)
                   [ L2 Cache & VRAM ]
```

- **Unified SRAM:** Shared memory and L1 cache live on the **exact same physical silicon** (64 KB–128 KB per SM).
- **Isolation:** Thread blocks on the same SM **cannot** see each other's shared memory or registers.

---

## 🚚 Memory performance: 32-byte transactions

- Global memory is accessed via **32-byte memory transactions**.
- When a thread requests a word, the warp **coalesces** the requests of all its threads into the number of transactions needed, depending on the **word size** and on how the **addresses are spread** across the threads.

> **If a thread requests a 4-byte word, the memory transaction the warp generates to global memory is 32 bytes in total.**

➡️ To use the memory system efficiently, the warp should **use all the bytes** fetched in a single transaction.

---

## ✅ Coalesced memory access

![w:860](img/w09-coalesced-memory-access.png)

<!-- _footer: "Figure: NVIDIA CUDA Programming Guide, Writing CUDA kernels: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html" -->

**A simple coalesced kernel** (C = A + B vector addition):
```python
@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]
```
- simple coalesced access

---

## ❌ Worst case: strided access

the pathologically worst case scenario is when consecutive threads access data elements that are 32 bytes or more apart from each other in memory. In this case, the warp will be forced to issue a 32-byte memory transaction for each thread, and the total number of bytes of memory traffic will be 32 bytes times 32 threads/warp = 1024 bytes.
![w:860](img/w09-coalesced-worst-case.png)

---

## 🔄 Matrix transpose in CUDA C

```C
/* macro to index a 1D memory array with 2D indices in row-major order */
/* ld is the leading dimension, i.e. the number of columns in the matrix     */

#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )

/* CUDA kernel for matrix transpose */
__global__ void cuda_transpose(int m, float *a, float *c ){
    int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    int myRow = blockDim.y * blockIdx.y + threadIdx.y;

    if( myRow < m && myCol < m ){
        c[INDX( myCol, myRow, m )] = a[INDX( myRow, myCol, m )];
    } /* end if */
    return;
} /* end cuda_transpose */
```

---

## 🔄 The transpose access pattern

![alt text](img/w09-matrix-transpose.png)

---

## 🔍 Which transpose access is coalesced?

<div class="cols">
<div class="card green">

### ✅ Read of `a`: coalesced
In a 2D thread block, `x` moves fastest, so consecutive `threadIdx.x` should access consecutive memory.

`threadIdx.x` appears in `myCol`, the **second** argument of `INDX`: consecutive threads read consecutive values of `a`.

</div>
<div class="card red">

### ❌ Write of `c`: not coalesced
`myCol` is the **first** argument of `INDX`: when it grows by 1, the address changes by `ld` (the leading dimension).

Consecutive threads write elements `ld` apart. When `ld` is larger than 32 (any matrix larger than 32), this is the **pathological case**.

</div>
</div>

💡 To alleviate the uncoalesced writes, use **shared memory** (next slides).

---

## 🏦 Shared memory banks

<div class="cols">
<div class="card blue">

### 🧱 How it is organised
- **32 banks**; successive 32-bit words map to successive banks
- each bank serves **32 bits per clock cycle**

⚠️ When threads of a warp access **different words in the same bank**, a **bank conflict** occurs: the accesses to that bank are **serialized**.

</div>
<div class="card green">

### ✅ Two exceptions
Several threads of a warp access the **same shared-memory location**:

- **read**: the word is **broadcast** to the requesting threads
- **write**: each address is written by only one of the threads (which one is undefined)

</div>
</div>

---

## 🏦 Bank access patterns

<div class="cols">
<div>

![h:500](img/w09-shared-memory-access.png)

</div>
<div class="small">

- **Left:** linear addressing with a stride of one 32-bit word (**no** bank conflict).
- **Middle:** linear addressing with a stride of two 32-bit words (**two-way** bank conflict).
- **Right:** linear addressing with a stride of three 32-bit words (**no** bank conflict).

</div>
</div>

<!-- _footer: "Figure: NVIDIA CUDA Programming Guide, Writing CUDA kernels: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html" -->

---

## 🔁 Transpose staged through shared memory

```python
import numpy as np
from numba import cuda
import cupy as cp

## Matrix transpose kernel, one thread per matrix element launched with 
## 2D thread block on 2D grid to match matrix size
## Stages input to shared memory
@cuda.jit
def smem_transpose(a, c):
    smemArray = cuda.shared.array(shape=(32, 32), dtype=np.float32)

    tile_col = cuda.blockDim.x * cuda.blockIdx.x
    tile_row = cuda.blockDim.y * cuda.blockIdx.y

    smemArray[(cuda.threadIdx.x, cuda.threadIdx.y)] = a[(tile_row + cuda.threadIdx.y, tile_col + cuda.threadIdx.x)]

    cuda.syncthreads()
    
    c[(tile_col + cuda.threadIdx.y, tile_row + cuda.threadIdx.x)] = smemArray[(cuda.threadIdx.y, cuda.threadIdx.x)]

```

---

## 🔁 What the tile copy does

-  Each thread in the same thread block has the same tileRow and tileCol values,
-  The kernel then proceeds with each thread block copying a 32 x 32 tile of the matrix from global memory to shared memory
   - Since the size of a warp is 32 threads, this copy operation will be executed by 32 warps, with no guaranteed order between the warps.

- Shared memory is used to stage data from global memory to ensure that reads from and writes to global memory are both coalesced properly.

- Shared memory is used to allow threads in the same thread block to share data among themselves.  

---

## 🩹 Padding removes bank conflicts

A common fix is to pad the shared memory by adding one to the column dimension:

`smemArray = cuda.shared.array(shape=(32, 32 + 1), dtype=np.float32)`

<div class="cols">
<div>

**Unpadded**

![h:250](img/w09-transpose-1.png)

</div>
<div>

**Padded (+1 column)**

![h:250](img/w09-transpose-2.png)

</div>
</div>

<span class="small">The numbers in the boxes indicate the warp index. The colors indicate which bank is associated with that shared memory location.</span>

---

<!-- _class: lead -->

# 📏 Counting the bytes

### Performance diagnosis

---

## 🐢 Your `vector_add` kernel: the memory wall

Your `vector_add` kernel (week08):

```python
c[i] = a[i] + b[i]      # 2 reads + 1 write = 12 bytes, 1 FLOP
```

- Arithmetic intensity: **1 / 12 ≈ 0.08 FLOP/byte**
- On a T4: 320 GB/s × 0.08 ≈ **27 GFLOP/s**, out of 8,100 GFLOP/s peak.

That was not a bug. That was the memory wall.

---

## 🩺 The only diagnostic you need first

$$\text{arithmetic intensity} = \frac{\text{FLOPs}}{\text{bytes moved from VRAM}}$$

T4: 8,100 GFLOP/s ÷ 320 GB/s ≈ **25 FLOP/byte** — the *ridge point*.

- Below 25 → **memory-bound**. More arithmetic units won't help.
- Above 25 → **compute-bound**. Now FLOPs matter.

---

## 🧩 Tile space

![h:400](img/w09-tile-vs-simt.png)

**Tile space:** conceptually partitioning an array into equally sized, non-overlapping tiles.

<!-- _footer: "Figure: NVIDIA CUDA Programming Guide, Programming Model: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html" -->

---

## 🧩 Tile space: how an array is divided

![h:410](img/w09-tile-space.png)

<span class="small">A two-dimensional array of shape (M, N). If a load operation specifies a tile shape of (tm, tn), the array is conceptually divided into rows and columns of tiles.</span>

<!-- _footer: "Figure: NVIDIA CUDA Programming Guide, Programming Model: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html" -->

---

![w:900](img/w09-roofline.svg)

<!-- _footer: "Course figure. Model ceilings, not measurements — you will add measured points in the lab." -->

---

## 🐢 Naive matmul: what does one thread read?

```python
col, row = cuda.grid(2)
acc = 0.0
for k in range(A.shape[1]):
    acc += A[row, k] * B[k, col]     # both from VRAM, every iteration
C[row, col] = acc
```

Each output reads a whole row of A and a whole column of B.

**Every element of A is read by N different threads.**

The accesses themselves are coalesced (see *Coalescing*); the problem here is **no reuse**.

---

![w:900](img/w09-reuse.svg)

---

<style scoped>section{font-size:25px} pre{font-size:16px}</style>

## 🧮 Example: 4×4 matmul with 2×2 tiles

Suppose we compute $C = A \times B$ for $N = 4$, using **2×2 thread blocks** ($T = 2$).
To compute the top-left 2×2 tile of $C$ (`C[0:2, 0:2]`):

```text
      A (4×4)                    B (4×4)                    C (4×4)
┌─────────┬─────────┐      ┌─────────┬─────────┐      ┌─────────┬─────────┐
│ Tile A0 │ Tile A1 │      │ Tile B0 │   ...   │      │ Block 0 │         │
├─────────┼─────────┤  ×   ├─────────┼─────────┤  =   ├─────────┼─────────┤
│   ...   │   ...   │      │ Tile B1 │   ...   │      │         │         │
└─────────┴─────────┘      └─────────┴─────────┘      └─────────┴─────────┘
```

**Step 1 (Phase 0):** The 4 threads in `Block 0` cooperate to load `Tile A0` and `Tile B0` (8 floats total) from VRAM into shared memory.
- Each thread computes its partial dot product from shared memory (using each loaded float **2 times**).

**Step 2 (Phase 1):** Synchronize, overwrite shared memory with `Tile A1` and `Tile B1` (8 floats), and add the second partial dot product.

---

Let's trace **Block (0, 0)** (4 threads) computing the top-left 2×2 output `C[0:2, 0:2]`:

```text
         A (4×4)                       B (4×4)                     C (4×4)
┌───────────┬───────────┐     ┌───────────┬───────────┐     ┌───────────┬───────────┐
│  1    2   │  3    4   │     │  2    0   │  ·    ·   │     │ C00  C01  │  ·    ·   │
│  5    6   │  7    8   │     │  1    3   │  ·    ·   │     │ C10  C11  │  ·    ·   │
├───────────┼───────────┤  ×  ├───────────┼───────────┤  =  ├───────────┼───────────┤
│  ·    ·   │  ·    ·   │     │  4    1   │  ·    ·   │     │  ·    ·   │  ·    ·   │
│  ·    ·   │  ·    ·   │     │  0    2   │  ·    ·   │     │  ·    ·   │  ·    ·   │
└───────────┴───────────┘     └───────────┴───────────┘     └───────────┴───────────┘
   Tile A0     Tile A1           Tile B0/B1                       Block (0,0)
```

Instead of each thread reading a full row and column from VRAM, the 4 threads march across A and down B **one 2×2 tile at a time**.

---

<style scoped>section{font-size:23px} pre{font-size:15px}</style>

## 🔢 Numerical walkthrough: Phase 0 & Phase 1

**Phase 0:** Each thread loads **1 element** of `Tile A0` and **1 element** of `Tile B0` into shared memory (`sA`, `sB`), then calls `cuda.syncthreads()`:

```text
sA = [ 1  2 ]      sB = [ 2  0 ]      Partial sums (from fast SRAM):
     [ 5  6 ]           [ 1  3 ]      • Thread (0,0): 1·2 + 2·1 = 4
                                      • Thread (0,1): 1·0 + 2·3 = 6
                                      • Thread (1,0): 5·2 + 6·1 = 16
                                      • Thread (1,1): 5·0 + 6·3 = 18
```
*(Notice: every number in `sA` and `sB` was fetched from VRAM **once**, but used by **2 threads**.)*

**Phase 1:** Synchronize, overwrite `sA` and `sB` with `Tile A1` and `Tile B1`, and accumulate:

```text
sA = [ 3  4 ]      sB = [ 4  1 ]      Final sums (partial + new):
     [ 7  8 ]           [ 0  2 ]      • Thread (0,0): 4  + (3·4 + 4·0) = 16
                                      • Thread (0,1): 6  + (3·1 + 4·2) = 17
                                      • Thread (1,0): 16 + (7·4 + 8·0) = 44
                                      • Thread (1,1): 18 + (7·1 + 8·2) = 41
```

---

## 💰 Counting the savings on this 4×4 example

Compare how many global memory reads **Block (0, 0)** performs to compute its 4 outputs:

| Approach | Reads per thread | Total VRAM reads for Block (0,0) | Reuse per float |
|---|---:|---:|---:|
| **Naive** (Week 8) | 4 from A + 4 from B = **8** | 4 threads × 8 = **32 floats** | **1×** (no reuse) |
| **Tiled ($T=2$)** | Loads 1 of A + 1 of B per phase | 2 phases × 8 = **16 floats** | **2×** ($T$ times) |

- **Scaling up:** A tile of size $T \times T$ cuts global VRAM traffic by **$T\times$**.
- For $N = 1024$ with $T = 16$, global reads drop from **8.59 GB** to **0.54 GB** — shrinking the memory bottleneck from **26.8 ms** down to **1.7 ms**.

---

## 🧾 Count it (float32, no reuse)

| N | GB read | GFLOP | FLOP/byte | ms @320 GB/s | ms of arithmetic |
|---:|---:|---:|---:|---:|---:|
| 512 | 1.07 | 0.27 | 0.25 | 3.4 | 0.03 |
| 1024 | 8.59 | 2.15 | 0.25 | 26.8 | 0.27 |
| 2048 | 68.7 | 17.2 | 0.25 | 214.7 | 2.12 |

The arithmetic needs **0.27 ms**. The bytes need **26.8 ms**.

📁 `week09/01_traffic_accounting.py` — no GPU needed.

---

## ⚠️ Two caveats

- This is a **no-reuse model**. The L1 and L2 caches catch some of the repeated reads, so the real naive kernel can be faster than 26.8 ms.
- The *shape* of the argument is right: moving the bytes takes ~100× longer than the arithmetic, so bandwidth, not FLOPs, is the limit.

---

## 🔧 The fix, in one sentence

> Load a piece of A and a piece of B into **fast on-chip memory once**,
> and let many threads reuse them.

That is **tiling**. 

---

## ❓ The four questions, refined

| Question | Week 8 | Today |
|---|---|---|
| **Split** | one thread per element | one **block per tile**, threads cooperate |
| **Assign** | block → any SM | block → one SM, **and its shared memory lives there** |
| **Move** | host ↔ VRAM | VRAM → **shared** → registers, once per tile |
| **Wait** | `cuda.synchronize()` (host) | `cuda.syncthreads()` — Week 3's barrier, in hardware |

---

## 🪜 A new rung on the cost ladder

```text
registers        ~1 cycle          ← the compiler manages these
shared memory    ~20–30 cycles     ← NEW: you manage this
L2               ~200 cycles
global (VRAM)    ~400–800 cycles
host RAM         PCIe, ~12 GB/s
network          ~ms               (Week 5)

approximate latencies, T4-class GPU
```

Week 5: *move computation to the data.*
Week 9: *move the data to fast memory, once, and reuse it.*

---

<!-- _class: lead -->

# 🏛️ The memory hierarchy

---

![w:880](img/w09-memory-hierarchy.svg)

---

## 🗺️ The spaces, by scope

| Space | Visible to | Lives as long as |
|---|---|---|
| **Registers** | one thread | the thread |
| **Local** | one thread | the thread |
| **Shared** | one block | the block |
| **Global (VRAM)** | grid + host | the allocation |
| **Constant** | grid, read-only | the allocation |

Scope is not a detail. It decides what can cooperate with what.

---

## 👷 Who manages it

| Managed by hardware | Managed by **you** |
|---|---|
| registers (compiler) | shared memory |
| L1 / L2 caches | global allocations |
| constant cache | host ↔ device transfers |

> Shared memory is a **user-managed cache**: you decide what to keep in it.
> (CUDA C++ Best Practices Guide, *Shared Memory*.)

---

<style scoped>section{font-size:24px}</style>

## 🏷️ "local" means scope, not location

In CUDA, **"local"** describes *who can see it* (private to one thread), **not** *where it sits on the chip*:

- **Registers (fast, ~1 cycle):** On-chip per-thread memory. Used for normal variables (`acc = 0.0`).
- **Local Memory (slow, ~400 cycles):** Off-chip **VRAM** reserved privately for one thread.

**When does slow local memory happen?**
1. **Register spilling:** Your thread uses too many variables, so excess registers overflow into VRAM.
2. **Dynamic indexing:** You create a small per-thread array (`cuda.local.array`) and index it with a variable (`arr[i]`) instead of a constant (`arr[0]`).

*Check it:* `kernel_resources()` in `gpu_utils.py` prints `spill bytes` — any number above `0` means hidden VRAM traffic.

---

## 💎 Why shared memory is worth the trouble

Every SM has its own fast on-chip shared memory, so bandwidth scales across all SMs in parallel:

| Memory Path | Where it lives | Aggregate Bandwidth (T4) |
|---|---|---|
| **Global / Local** | Off-chip VRAM | **~320 GB/s** (shared by whole GPU) |
| **Shared Memory** | On-chip SRAM (inside each SM) | **~8,140 GB/s (~8 TB/s)** |

**Where does ~8 TB/s come from?**
- **Per SM:** 32 banks × 4 bytes/cycle × 1.59 GHz clock ≈ **203.5 GB/s**
- **Across 40 SMs:** 203.5 GB/s × 40 SMs ≈ **8,140 GB/s**

That is **~25× more bandwidth** than VRAM—*provided* threads don't stall on bank conflicts.

---

## 🔗 Scope binds scheduling

- A block's shared memory is allocated on **one SM**.
- Therefore a block cannot migrate, and `cuda.syncthreads()` can only
  synchronise **within that block**.
- Across blocks: still no cooperation inside one launch (Week 8's contract).

Need a grid-wide boundary? **Launch a second kernel.**

---

<!-- _class: lead -->

# 🧱 Coalescing

---

## 📦 What the hardware actually fetches

- Global memory is served in **aligned 32-byte sectors** (four of them make a 128-byte cache line).
- A warp's 32 requests are merged into as few sectors as possible.
- Bytes fetched but not used are **wasted bandwidth**.

If every lane's 4-byte access needs its own 32-byte sector, only 4 of every 32 bytes
fetched are useful: **8× less effective bandwidth**.

---

![w:900](img/w09-coalescing.svg)

---

## 🗑️ The arithmetic of waste

**Coalesced** — lane *i* reads `a[i]`:
32 lanes × 4 B = 128 B, in 4 sectors of 32 B. **100 % used.**

**Strided** — lane *i* reads `a[i*1024]`:
32 separate sectors × 32 B = 1024 B fetched, 128 B used. **8× waste.**

Same instructions. Same useful bytes. Eight times the traffic.

---

## 🔮 Predict before we run

A 4096 × 4096 float32 matrix (64 MiB), stored row-major.

- Kernel A: each thread sums **one row**.
- Kernel B: each thread sums **one column**.

Same FLOPs. Same total bytes.

**Will they take the same time?** Write down your guess and a factor.

---

## 🐌 Kernel A: one thread per row

```python
@cuda.jit
def row_sums(A, out):
    row = cuda.grid(1)
    if row < A.shape[0]:
        acc = 0.0
        for c in range(A.shape[1]):
            acc += A[row, c]        # lanes i and i+1 are N floats apart
        out[row] = acc
```

Each thread walks along *its own* row, which feels natural for a row-major matrix.
But at step `c`, consecutive lanes read `A[i, c]`, `A[i+1, c]`, ... : **N floats apart**.

---

## 🚀 Kernel B: one thread per column

```python
@cuda.jit
def col_sums(A, out):
    col = cuda.grid(1)
    if col < A.shape[1]:
        acc = 0.0
        for r in range(A.shape[0]):
            acc += A[r, col]        # lanes i and i+1 are adjacent in memory
        out[col] = acc
```

At step `r`, consecutive lanes read `A[r, j]`, `A[r, j+1]`, ... : **consecutive addresses**.

The thread that *looks* less natural is the fast one.

---

## ▶️ Run it

```bash
python 02_coalescing.py            # prints ms, achieved GB/s and % of peak for both kernels
```

- Both kernels are verified against NumPy's `A.sum(axis=...)` before they are timed.
- **Achieved GB/s = bytes touched ÷ time.** A measurement, not a model.
- Expect `col_sums` to sit much closer to the 320 GB/s ceiling. The 8× of the sector model
  is an upper bound: L1/L2 caches soften it, so the measured gap depends on the GPU.

📁 `week09/02_coalescing.py`

---

## 🤔 The surprise: almost the same time

**The prediction:** `col_sums` (coalesced) should beat `row_sums` (strided) by a clear margin.

On a T4, `n = 4096`: **almost the same time.** Sometimes within 10 %.

**Was the model wrong?**

- The *model* describes the number of 32-byte sectors a warp requests **on a miss**.
- Whether that miss costs bandwidth depends on the **cache**.

Reads-with-reuse are the friendliest case for the memory system.
Modern GPUs are very good at the friendly case.
> Coalescing matters when the wasted bytes are never re-used. It doesn't matter when the cache will re-use them for you.

---

## 🧬 The same lesson, different spelling: AoS vs SoA

```python
# Array of Structs: stride between lanes
particles[i].x        # x, y, z, x, y, z, ...

# Struct of Arrays: contiguous per field
x[i]                  # x, x, x, x, ...
```

Deep learning already prefers SoA: a tensor is one contiguous field,
not a list of records. Now you know why.

---

## 📚 The natural C layout: Array of Structs

```c
struct Particle {
    float x, y, z;            // 12 bytes
};
struct Particle particles[N];
```
```
addr:   0    4    8   12   16   20   24   28  ...
        x0   y0   z0   x1   y1   z1   x2   y2  ...
        └────particle 0────┘└────particle 1───┘
```

---

## 🧱 The other layout: Struct of Arrays

```c
float x[N], y[N], z[N];
```
```
x:   x0   x1   x2   x3   x4   x5   ...    (contiguous)
y:   y0   y1   y2   y3   y4   y5   ...    (contiguous)
z:   z0   z1   z2   z3   z4   z5   ...    (contiguous)
```
One contiguous field per array. Access is `x[i], y[i], z[i]`.

> Same number of bytes. Same arithmetic. Only the layout differs.

---

## 📏 The stride between lanes

Almost every GPU kernel is *"`thread-i` handles `element-i`."*
Suppose each thread only needs the **x** field:

```c
i = cuda.grid(1)
val[i] = particles[i].x           # AoS
```
Lane `i` reads address `i × 12` bytes. **Consecutive lanes are 12 bytes apart**.

A warp (32 lanes) touches addresses `0, 12, 24, …, 372`:

| | |
|---|---:|
| useful bytes (32 × 4 B) | 128 B |
| cachelines touched | ~3 × 128 B = **384 B** |
| bytes fetched / useful | **≈ 3×** |

---

## ✅ SoA: the same warp, one cacheline

```c
val[i] = x[i]                     # SoA
```

Lane `i` reads address `i × 4` bytes. **Consecutive lanes are 4 bytes apart.**

Addresses `0, 4, 8, …, 124` → **one 128-byte cacheline.**

| | AoS `particles[i].x` | SoA `x[i]` |
|---|---:|---:|
| stride between lanes | 12 B | **4 B** |
| bytes fetched | ~384 B | **128 B** |
| useful fraction | ~33 % | **100 %** |
| verdict | strided ❌ | **coalesced** ✅ |

This is *exactly* the coalescing rule again — the lane-to-address map, nothing else.

---

## 🤷 When AoS is fine

AoS is only wasteful when threads read **one field at a time**.

If every thread genuinely needs all three fields *together*:

```cuda
v = sqrt(particles[i].x * particles[i].x
       + particles[i].y * particles[i].y
       + particles[i].z * particles[i].z)     # touches every byte
```

then AoS uses every byte it fetches — **no waste.**


> SoA wins when the access pattern is **per-field**, 
>   - which is the overwhelming case in GPU kernels, DL, and physics simulations.

---

## 🔥 AoS vs SoA is already in your tensors


A `(N, C, H, W)` tensor is not a list of records. It is a stack of
**contiguous per-field** planes:

```text
channel 0:  [ ... N×H×W floats, contiguous ... ]
channel 1:  [ ... N×H×W floats, contiguous ... ]
...
```

The **innermost dimension is contiguous** (`W`). Every conv / matmul / LayerNorm kernel processes one feature dimension at a time — the "read one field per lane" pattern.

A framework that stored images AoS-style (`[pixel: (r,g,b), …]`) would make every per-channel kernel strided. They do not. That is not an accident.

---

## 🏅 Week 8's convention, vindicated

```python
col, row = cuda.grid(2)      # x → columns, y → rows
```

We chose `x → columns` in Week 8 for a reason we could not yet explain:
consecutive `threadIdx.x` then touch consecutive memory.

**That choice was coalescing.** Swap the roles and measure the cost.

---

<style scoped>section{font-size:25px} table{font-size:20px}</style>

## 🧩 The unifying table


> **The layout of data in memory and the map of threads to that memory
> decide whether a warp's 32 lanes collapse to 1 transaction.**

| Situation | Layout | Thread → element | Result |
|---|---|---|---|
| `col_sums` | row-major matrix | lane → consecutive columns | ✅ |
| `row_sums` | row-major matrix | lane → consecutive rows (stride N) | ❌ |
| **SoA** | `x[N], y[N], z[N]` | lane → `x[i]` (stride 4 B) | ✅ |
| **AoS** | `[(x,y,z), …]` | lane → `particles[i].x` (stride 12 B) | ❌ |
| Week 8 `col, row` | row-major | `threadIdx.x` → column | ✅ |
| Week 8 `row, col` | row-major | `threadIdx.x` → row (stride N) | ❌ |

**Two levers, one bottleneck:** change the thread map (Week 8) **or** the memory layout (AoS → SoA). Real code often pulls both.

---

<!-- _class: lead -->

# 🧩 Shared memory and tiling

---

## 📝 The idea, on paper first

One block computes a **T × T tile** of C. It walks along K in steps of T:

1. **Load** a T×T tile of A and of B into shared memory (one element per thread).
2. `cuda.syncthreads()` — the tile is complete.
3. **Compute** T multiply-adds per thread, all from shared memory.
4. `cuda.syncthreads()` — nobody overwrites the tile early.

Each loaded value is used **T times**.

---

![w:880](img/w09-tiling.svg)

---

## 📉 What that buys: traffic ÷ T

Global reads: $2N^3 \rightarrow \dfrac{2N^3}{T}$ &nbsp;&nbsp;·&nbsp;&nbsp; intensity: $0.25 \rightarrow \dfrac{T}{4}$

| T | GB read (N=1024) | FLOP/byte | shared / block |
|---:|---:|---:|---:|
| 1 (naive) | 8.59 | 0.25 | — |
| 8 | 1.07 | 2.0 | 0.5 KB |
| 16 | 0.54 | 4.0 | 2 KB |
| 32 | 0.27 | 8.0 | 8 KB |

---

![w:820](img/w09-traffic.svg)

Even T=32 is below the ridge of 25. **cuBLAS adds register blocking on top.**

---

## 🚧 The two barriers are not optional

<div class="cols">
<div class="card red">

### ❌ No barrier after the **load**
A thread reads a tile element **nobody has written yet**.

</div>
<div class="card red">

### ❌ No barrier after the **compute**
A fast thread **overwrites the tile** while a slow one still reads it.

</div>
</div>

🔁 This is Week 3's **race condition**, on silicon.

We will remove each one and look at the answer.

---

## 📜 Barrier rules (from Week 8, now load-bearing)

```python
for k0 in range(0, K, T):
    sA[ty, tx] = ...; sB[ty, tx] = ...
    cuda.syncthreads()               # every thread reaches this
    for k in range(T): ...
    cuda.syncthreads()               # and this
```

- **Every** thread in the block must reach the barrier.
- Never put it inside `if i < n:` or inside a divergent branch.
- It synchronises a **block**, never the grid.

---

## 🧱 Numba: declaring shared memory

```python
from numba import cuda, float32

@cuda.jit
def kernel(A, B, C):
    sA = cuda.shared.array(shape=(16, 16), dtype=float32)
    sB = cuda.shared.array(shape=(16, 16), dtype=float32)
```

- Allocated per **block**, on chip, uninitialized.
- `shape` must be a **compile-time constant** (see the Numba CUDA memory docs).
- `cuda.local.array(...)` is the per-thread version; it lives in *local memory*, which is backed by VRAM.

---

## 🏭 Consequence: to sweep T, use a factory

```python
def make_tiled(T):                      # T becomes a compile-time constant
    @cuda.jit
    def kernel(A, B, C):
        sA = cuda.shared.array(shape=(T, T), dtype=float32)
        ...
    return kernel

k16 = make_tiled(16)                    # two distinct compiled kernels
k32 = make_tiled(32)
```

You cannot pass the tile size as a runtime argument. This is the first thing you will hit.

---

## 🧑‍💻 The kernel, part 1: who am I, and load

```python
def make_tiled(T):
    @cuda.jit
    def kernel(A, B, C):
        sA = cuda.shared.array(shape=(T, T), dtype=float32)
        sB = cuda.shared.array(shape=(T, T), dtype=float32)
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        bx, by = cuda.blockIdx.x, cuda.blockIdx.y
        row, col = by * T + ty, bx * T + tx
        acc = 0.0                                  # a register, per thread
        for k0 in range(0, A.shape[1], T):
            sA[ty, tx] = A[row, k0 + tx]           # one element per thread
            sB[ty, tx] = B[k0 + ty, col]
            cuda.syncthreads()                     # (1)
```

---

## 🧑‍💻 The kernel, part 2: compute and store

```python
            for k in range(T):
                acc += sA[ty, k] * sB[k, tx]       # T reuses, all on chip
            cuda.syncthreads()                     # (2)
        C[row, col] = acc
    return kernel
```

Launch (N a multiple of T, so no bounds checks yet):

```python
k = make_tiled(16)
k[(N // 16, N // 16), (16, 16)](dA, dB, dC)
cuda.synchronize()
assert np.allclose(dC.copy_to_host(), A @ B, atol=1e-2)
```

📁 `week09/04_matmul_tiled.py` (it also takes a `drop_barrier` option, used in the race demo below)

---

## 🔮 Predict, then run

At N = 1024 the naive kernel must read 8.6 GB: at least **27 ms** at 320 GB/s (see *Counting the bytes*).

Tiling with T = 16 cuts global traffic by **16×**.

- Will the kernel be 16× faster?
- What else could limit it now?

Write a number down. Then run it.

---

## ▶️ Run it

```bash
python 03_matmul_naive.py          # naive kernel, N = 1024
python 04_matmul_tiled.py          # tiled T = 16, then the same N with cuBLAS via CuPy
python 05_tile_sweep.py            # naive, T = 8, 16, 32 and cuBLAS for N = 512, 1024, 2048
```

Each line reports ms, GFLOP/s and the max error against `A @ B`.

Your tiled kernel will **not** beat cuBLAS. It should beat your naive kernel
by a clear margin. That is the win.

📁 `week09/03_matmul_naive.py`, `week09/05_tile_sweep.py`

---

## 💥 Now remove a barrier

```bash
python 04_matmul_tiled.py --race
```

The script runs each broken variant three times and prints the max error:

- drop the barrier **after the load**: threads may read tile elements nobody has written yet;
- drop the barrier **after the compute**: a fast thread may overwrite the tile while a slow one still reads it.

The error can differ from run to run, and sometimes it is **zero**.
A race is not a crash: it is a *nondeterministic* answer. That is what makes it dangerous.

---

## 🏦 Shared memory is banked

- 32 **banks**, 4 bytes wide, accessible in parallel.
- *n* addresses in *n* distinct banks → served together.
- Two addresses in the same bank → **serialized**: an *n*-way bank conflict.
- Exception: all lanes reading the **same address** → one **broadcast**, no conflict.

---

![w:860](img/w09-banks.svg)

---

<style scoped>section{font-size:24px} table{font-size:20px}</style>

## 🧮 Banks of a 2D shared array

Shared memory is **32 banks × 4 bytes**. Word `w` lives in bank `w mod 32`.
The hardware serves **one word per bank per cycle**.

For a row-major `tile[R][C]`:

$$\text{bank}(tile[r][c]) = (r \cdot C + c) \bmod 32$$

The **row stride is the column count `C`**. That one number decides
whether the row index helps or vanishes:

| `C` | `C mod 32` | row index contributes? |
|---:|---:|---|
| 32 | 0 | **no** — bank = `c mod 32` |
| 33 | 1 | yes — every row shifts the banks by 1 |
| 34 | 2 | partly — only 16 distinct banks for 32 rows |
| 31 | 31 | yes, and coprime → sweeps all 32 |

---

## 🚦 The three cases

A warp is 32 lanes. What they touch decides the cost:

| Pattern across the warp | Cost |
|---|---|
| 32 lanes → **32 distinct banks**, different words | **1** transaction ✅ |
| 32 lanes → **the same word** | **1** transaction (broadcast) ✅ |
| *n* lanes → **the same bank**, *n* different words | **n** serialized transactions ❌ |

Two of these are free. Only the third is a **bank conflict**.

Note the middle case: reading the *same address* from every lane is a
broadcast, not a conflict. That is what saves the matmul kernel.

---

## ✅ Important: our matmul does *not* conflict

In the inner loop, take a warp with `tx = 0..31` (T = 32, so one value of `ty`):

| access | addresses across the warp | result |
|---|---|---|
| `sA[ty, k]` | all the **same** address | **broadcast**, 1 request |
| `sB[k, tx]` | bank = `tx`, all distinct | conflict-free |

With T = 16 a warp spans two rows of the tile: two addresses 16 words apart, in different banks. Still no conflict.

The 32-way conflict belongs to the **transpose** pattern `tile[tx, ty]`.
Let's demonstrate it there.

---

## 🔎 The transpose: which access is the problem

Block is `(T, T) = (32, 32)` = 1024 threads. Global reads and writes
are both coalesced. The shared-memory access is where it hurts.

```python
T = 32
@cuda.jit
def kernel(A, out):
    tile = cuda.shared.array(shape=(T, T), dtype=float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tile[ty, tx] = A[blockIdx.y*T + ty, blockIdx.x*T + tx]   # write
    cuda.syncthreads()
    out[blockIdx.x*T + ty, blockIdx.y*T + tx] = tile[tx, ty]  # READ ← strided
```

Two shared-memory accesses. They behave differently:

| access | across a warp (fixed `ty`, `tx` = 0…31) |
|---|---|
| **write** `tile[ty][tx]` | `tx` is the **column** index → sweeps banks, conflict-free |
| **read** `tile[tx][ty]` | `tx` is the **row** index → multiplies the stride |

---

## 🧵 Warp layout: consecutive `tx`, one `ty`

A warp is 32 **consecutive linear thread ids**. With `blockDim = (32, 32)`:

$$\text{tid} = ty \cdot 32 + tx$$

So a warp is **32 consecutive `tx`, all with the same `ty`** (warp `w` ⇔ `ty = w`).

That is the layout of the transpose read:

```text
lane:  0    1    2    3   …   31
tx:    0    1    2    3   …   31
ty:    w    w    w    w   …    w      (same for all lanes)
reads: tile[0][w], tile[1][w], tile[2][w], …, tile[31][w]
```

**Column `w` of the tile.** Every lane touches a different **row**, same **column**.
That is the strided pattern — and shared memory charges for it, unlike VRAM.

---

<style scoped>section{font-size:23px} table{font-size:18px} td,th{padding:2px 8px}</style>

## ❌ Unpadded (32): the `tx` term vanishes

`width = 32`. The read is `tile[tx][ty]`, word index `tx·32 + ty`:

$$\text{bank} = (tx \cdot 32 + ty) \bmod 32 = \underbrace{(tx \cdot 32) \bmod 32}_{=\,0} + ty = ty$$

The `tx` dependence **cancels** — every lane in the warp computes the same bank.

| lane `tx` | reads | word index | bank |
|---:|---|---:|---:|
| 0 | `tile[0][ty]` | `0·32 + ty` | `ty` |
| 1 | `tile[1][ty]` | `1·32 + ty` | `ty` |
| 2 | `tile[2][ty]` | `2·32 + ty` | `ty` |
| … | … | … | `ty` |
| 31 | `tile[31][ty]` | `31·32 + ty` | `ty` |

**1 distinct bank for 32 lanes, and all different words → 32-way conflict → 32 serialized cycles** for one transaction.

---

<style scoped>section{font-size:23px} table{font-size:18px} td,th{padding:2px 8px}</style>

## ✅ Padded (33): the `tx` term returns

`width = 33`. Word index `tx·33 + ty`:

$$\text{bank} = (tx \cdot 33 + ty) \bmod 32 = \big(tx \cdot \underbrace{(33 \bmod 32)}_{=\,1} + ty\big) \bmod 32 = (tx + ty) \bmod 32$$

`tx` now contributes **linearly**, and `tx ↦ (tx + ty) mod 32` is a **rotation** of `Z₃₂` — a bijection. With `ty = 0`:

| lane `tx` | word index | bank |
|---:|---:|---:|
| 0 | `0` | `0` |
| 1 | `33` | `1` |
| 2 | `66` | `2` |
| … | … | … |
| 31 | `1023` | `31` |

**32 distinct banks → 1 transaction.** The rotation by `ty` is irrelevant; bijection is all that matters. (The **write** `tile[ty][tx]` was already conflict-free in both versions — see the table on the "which access" slide.)

---

<style scoped>section{font-size:23px} table{font-size:19px}</style>

## 🍀 Why 33 — and why 32 is the unlucky number

The requirement is just: **`width` coprime to 32**, i.e. `width` odd.
- Then `tx ↦ tx·width mod 32` is a bijection on `Z₃₂`.

`width = 32` is the **worst** case because `32 ≡ 0 (mod 32)`:
- the row index **disappears** from the bank formula, so a column-strided
access (all lanes in one column) collapses to a single bank.

`width = 33` is the **smallest** odd stride, so it wastes the least shared memory:

| version | bank of lane `tx` | distinct banks | shared/block |
|---|---|---:|---:|
| `(T, T)`, T=32 | `(tx·32 + ty) % 32 = ty` | **1** | 4.00 KB |
| `(T, T+1)` = 33 | `(tx·33 + ty) % 32` | **32** | 4.12 KB |

**128 extra bytes** (one float × 32 rows) remove a **32-way serialization**
on the hot access. That is the trade.

📁 `week09/06_bank_conflicts.py` — both versions verified against `A.T`.

---

## 📌 One rule, one sentence

> **A warp is free when its lanes land in distinct banks (different words)
> or on the same word (broadcast).**
>
> **A row-major 2D array with a multiple-of-32 row stride makes the row
> index vanish from the bank formula — so a column-strided access
> collapses to one bank.**
>
> **Pad the stride to the next odd number.**

That single change — 32 → 33 — turns a 32-cycle serialization into 1 cycle
for 3 % more shared memory. 
The tiled matmul,the block reduction, the softmax are already bank-friendly; the transpose is the one kernel where the natural index order fights the bank layout.

---

## 🏁 Three-way summary

| version | traffic | why it is where it is |
|---|---|---|
| naive | 2N³ reads | no reuse: bandwidth-bound at 0.25 FLOP/byte |
| tiled T=32 | 2N³/32 | reuse in shared memory: 8 FLOP/byte |
| cuBLAS | lower still | larger tiles + register blocking + per-architecture tuning |

Use the library for matmul. Write kernels for what the library does not have.

---

<!-- _class: lead -->

# ➕ Reductions: softmax redeemed

---

## 🤦 What was wrong with Week 8's softmax

```python
# Week 8, deliberately naive: EVERY thread did this
for j in range(n): row_max = max(row_max, logits[row, j])
for j in range(n): row_sum += exp(logits[row, j] - row_max)
```

Correct, but each of the *n* threads recomputed the whole row: **n× redundant work**.

The row max and row sum are **reductions**. Threads must cooperate.

---

## 🌳 Tree reduction: same work, shorter chain

| | work | span (critical path) |
|---|---|---|
| one thread sums N | N−1 adds | N−1 |
| tree over N lanes | N−1 adds | **log₂N** |

Week 8's work-and-span slide, now implemented.

Requires an **associative** operator. Floating-point sums may differ in the
last digits when the order changes — that is expected, not a bug.

---

![w:860](img/w09-reduction.svg)

---

## 🧑‍💻 Code: block sum in shared memory

```python
s = cuda.shared.array(shape=TPB, dtype=float32)
tid = cuda.threadIdx.x
i = cuda.grid(1)
s[tid] = x[i] if i < x.size else 0.0      # pad the tail with the identity
cuda.syncthreads()
stride = TPB // 2
while stride > 0:
    if tid < stride:
        s[tid] += s[tid + stride]
    cuda.syncthreads()                    # OUTSIDE the if
    stride //= 2
# s[0] holds this block's total
```

---

## 🔑 Two details that matter

<div class="cols">
<div class="card blue">

### `tid < stride`
Not `tid % (2*stride) == 0`.

**Contiguous** active lanes: whole warps go idle instead of diverging. Same answer, better warp efficiency.

</div>
<div class="card orange">

### The barrier is outside the `if`
**Every** thread must arrive, including the ones doing nothing this round.

Putting it inside the `if` is the classic **deadlock / undefined-behaviour** bug.

</div>
</div>

---

## 🧱 Across blocks: two options

| option | how | cost |
|---|---|---|
| **atomic** | `cuda.atomic.add(total, 0, s[0])` once per block | serializes on one address |
| **second kernel** | write partials, launch a reduce kernel | an extra launch + a grid boundary |

Never one atomic **per thread**: that is Week 3's `sum += 1` race, "fixed"
by making every thread queue on one lock.

---

## ⚛️ One atomic per block vs one per thread

| version | atomics issued (n = 2²⁴) |
|---|---:|
| shared tree + 1 atomic per block (256 threads) | 65,536 |
| atomic per thread | 16,777,216 |

```bash
python 07_block_reduce.py          # ms, GB/s, result and error against a float64 sum, for both
```

📁 `week09/07_block_reduce.py`

Both results are "correct", and their last digits differ. Explain why.

---

## 🔁 Softmax: three passes over shared memory

```text
one block per row, TPB threads (TPB ≥ n_cols, power of two)

pass 1   load row → shared, tree-reduce MAX      → row_max
pass 2   e = exp(val − row_max) → shared, tree-reduce SUM → row_sum
pass 3   out = e / row_sum
```

Padding lanes contribute `-1e30` to the max and `0.0` to the sum.

---

## 🧑‍💻 Code: pass 1 (max)

```python
val = logits[row, tid] if tid < n else -1e30
s[tid] = val
cuda.syncthreads()
stride = TPB // 2
while stride > 0:
    if tid < stride:
        s[tid] = max(s[tid], s[tid + stride])
    cuda.syncthreads()
    stride //= 2
row_max = s[0]                 # every thread reads it...
cuda.syncthreads()             # ...before anyone overwrites s
```

That extra barrier is easy to forget. Without it, pass 2 races pass 1.

---

## 🧑‍💻 Code: passes 2 and 3

```python
e = math.exp(val - row_max) if tid < n else 0.0
s[tid] = e
cuda.syncthreads()
stride = TPB // 2
while stride > 0:
    if tid < stride:
        s[tid] += s[tid + stride]
    cuda.syncthreads()
    stride //= 2
row_sum = s[0]
cuda.syncthreads()
if tid < n:
    out[row, tid] = e / row_sum     # e kept in a register: no re-read
```

📁 `week09/08_softmax_shared.py` — verified against `scipy.special.softmax`.

---

## ▶️ Run it

| rows × cols | TPB | what it tests |
|---|---|---|
| 4096 × 64 | 64 | one thread per column |
| 4096 × 512 | 512 | a full-size block per row |
| 1024 × 37 | 64 | the padding path: 37 columns, 64 threads |

The script checks every case against `scipy.special.softmax` and prints ms, max error
and the mean row sum (it must be 1.000000).

Still slower than cuDNN for wide rows: that needs warp shuffles and more tiling.

---

## 🎁 Optional: the last 32 lanes need no shared memory

```python
# Numba exposes the warp shuffle:
#   cuda.shfl_down_sync(membermask, value, delta)
val += cuda.shfl_down_sync(0xffffffff, val, 16)
val += cuda.shfl_down_sync(0xffffffff, val, 8)   # ... 4, 2, 1
```

Within a warp, lanes exchange **registers** — no shared memory, no barrier.
Name it, recognise it in papers; implement it as a bonus exercise.

---

<!-- _class: lead -->


# 📊 Occupancy

---

## 📖 Definition

$$\text{occupancy} = \frac{\text{resident warps per SM}}{\text{maximum warps per SM}}$$

Why it matters: the SM hides memory latency by switching to another
**ready warp**. Few resident warps → nothing to switch to → stalls.

It is a *latency-hiding proxy*, not a performance target.

---

## 📏 T4 (sm_75) limits

| resource | per SM |
|---|---:|
| threads | 1,024 |
| warps | 32 |
| blocks | 16 |
| shared memory | 64 KB |
| 32-bit registers | 65,536 |

Read them live: `gpu_utils.device_report()`.

---

## 🧮 Worked calculation: our own kernels

| kernel | threads/block | shared/block | limiting resource | blocks/SM | occupancy |
|---|---:|---:|---|---:|---:|
| tiled T=8 | 64 | 0.5 KB | blocks (16) | 16 | 100 % |
| tiled T=16 | 256 | 2 KB | threads | 4 | 100 % |
| tiled T=32 | 1024 | 8 KB | threads | 1 | 100 % |

(Assuming at most 64 registers per thread.) All three can reach 100 %, **and their runtimes still differ.**

---

![w:800](img/w09-occupancy.svg)

---

## 🧯 Registers usually bite first

- 256 threads × 96 registers = 24,576 → only **2** blocks fit in 65,536 → 50 %.
- Check the register count instead of guessing.

See it:

```python
from gpu_utils import kernel_resources
kernel_resources(k)        # regs/thread and spill bytes, per compiled signature
```

Spills (`local` bytes > 0) mean VRAM traffic you did not ask for.

---

<style scoped>section{font-size:24px} table{font-size:18px} td,th{padding:3px 8px}</style>

## 🔍 Walkthrough: what limits occupancy on a T4?

A T4 SM has three hard ceilings: **1,024 threads**, **64 KB shared memory**, and **65,536 registers**.
Suppose we launch blocks of **256 threads** (8 warps per block; 4 blocks = 100% occupancy):

| Kernel profile | Shared / block | Regs / thread | Regs / block | Bottleneck resource | Max blocks / SM | Achieved occupancy |
|---|---:|---:|---:|---|---:|---:|
| **A: Light** | 2 KB | 32 | 8,192 | **Threads** (4 × 256 = 1,024) | **4** | **100%** (32/32 warps) |
| **B: Shared-heavy** | 24 KB | 32 | 8,192 | **Shared mem** (2 × 24 KB = 48 KB) | **2** | **50%** (16/32 warps) |
| **C: Register-heavy** | 2 KB | 96 | 24,576 | **Registers** (2 × 24,576 = 49,152) | **2** | **50%** (16/32 warps) |

---

## 🔍 Why no third block fits

<div class="cols">
<div class="card orange">

### Case B: shared-heavy
A 3rd block needs **72 KB**, more than the **64 KB** limit.

</div>
<div class="card red">

### Case C: register-heavy
A 3rd block needs **73,728 registers**, more than the **65,536** limit.

</div>
</div>

💡 **Takeaway:** whichever resource hits its SM ceiling **first** caps the number of resident warps.

---

## 🎯 Occupancy is not the goal

- Volkov's classic result: **more work per thread** (instruction-level
  parallelism) can win at *lower* occupancy.
- Bigger tiles → less traffic, but fewer independent instructions per thread.
- Our T=8/16/32 sweep is exactly this experiment.

**Recipe:** sweep the tile size {8, 16, 32} (the block is T × T threads);
for a 1D kernel, sweep the block size {64, 128, 256, 512, 1024}.
Keep the fastest one that is **correct**.

---

<!-- _class: lead -->

# 🔬 Profiling

---

## 🛠️ Two tools, two different questions

| tool | question | scope |
|---|---|---|
| **Nsight Systems** (`nsys`) | where is the time going *overall*? | timeline: transfers, launches, gaps |
| **Nsight Compute** (`ncu`) | why is *this kernel* slow? | per-kernel counters |

Start with the timeline. A kernel that is 5 % of the runtime is not your problem.

---

## 📊 The first four metrics to read

| metric | tells you |
|---|---|
| DRAM throughput (% of peak) | are you bandwidth-bound? |
| compute throughput (% of peak) | are you compute-bound? |
| achieved occupancy | can the SM hide latency? |
| warp stall reasons | *what* is it waiting for? |

Everything else is detail until these four say something.

---

## ⚠️ Colab reality check

- `ncu` is often **not installed** on Colab.
- Even installed, counters are frequently blocked:
  `==ERROR== ERR_NVGPUCTRPERM` — the user lacks permission for GPU performance counters.
- `nvprof` is **deprecated** and replaced by the Nsight tools; do not build on it.

So: try the tool, then fall back. The fallback is real profiling.

---

## 🧪 Try it

```bash
which ncu                      # is Nsight Compute there at all?
ncu --set basic --launch-count 1 --target-processes all python 03_matmul_naive.py
```

Nsight Compute ships with the CUDA Toolkit; on Colab it is usually missing.

📁 `week09/09_profile_and_measure.py` tries `ncu`, detects `ERR_NVGPUCTRPERM`,
and falls back to measurement automatically.

---

## 🛟 The fallback that always works

$$\text{achieved GB/s} = \frac{\text{bytes moved}}{\text{time}} \qquad
\text{achieved GFLOP/s} = \frac{2N^3}{\text{time}}$$

Compare with 320 GB/s and 8,100 GFLOP/s. Whichever ceiling you are near
**is your bottleneck** — and you can state it with a number.

Plus: sweep a parameter and watch the time. That is an experiment.

---

## ⏱️ Timing a kernel correctly

```python
start, end = cuda.event(), cuda.event()   # events record timing by default
start.record(); fn(); end.record()
end.synchronize()                         # launches are asynchronous: wait for `end`
ms = start.elapsed_time(end)              # same as cuda.event_elapsed_time(start, end)
```

- Warm up the JIT first: the first launch includes compilation.
- Report the **median** of at least 5 runs, not one sample.
- `time_ms(fn)` in `gpu_utils.py` does all of this.

---

## 📈 `torch.profiler`: the Week 10 preview

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        a @ b
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=6))
prof.export_chrome_trace("week09_torch_trace.json")
```

Open the JSON in `chrome://tracing` or `ui.perfetto.dev`.
**Gaps between kernels = the GPU is idle.** Week 10 is about closing them.

---

<!-- _class: lead -->

# 🌊 Streams: overlapping copies and compute

---

## ⏳ The GPU waits during copies

So far every program did the same three things, one after another:

```text
upload  →  kernel  →  download
```

- During the **copy**, the SMs are idle. During the **kernel**, the copy engines are idle.
- PCIe is ~12 GB/s (Week 6): for a light kernel, the copies can cost more than the compute.

Idea: **split the data into chunks and pipeline them**, like a factory line.

---

![w:900](img/w09-streams.svg)

---

## 🌊 What is a stream?

A **stream** is a queue of GPU work: copies and kernel launches.

- Operations in **one** stream run **in order**.
- Operations in **different** streams **may** overlap. Nothing forces them to.
- Everything you have launched so far went into the **default stream**.

```python
s = cuda.stream()                         # create a queue
d_in = cuda.to_device(h_chunk, stream=s)  # async upload into s
kernel[blocks, tpb, s](d_in, d_out)       # launch into s (third entry of the launch config)
d_out.copy_to_host(h_out_chunk, stream=s) # async download into s
s.synchronize()                           # wait for this stream; cuda.synchronize() waits for all
```

---

## ❓ The four questions, once more

| Question | With streams |
|---|---|
| **Split** | the array into K independent chunks |
| **Assign** | each chunk to its own stream |
| **Move** | async copies, in and out, queued in that stream |
| **Wait** | `cuda.synchronize()` once, after every chunk is queued |

Week 5's lesson again: *overlap communication with computation.*

---

## 🌊 Three requirements for overlap

<div class="cols3">
<div class="card green">

### 📌 Pinned host memory
`cuda.pinned_array(n, dtype)` or `with cuda.pinned(a, b):`

Copies from pageable memory are staged and **cannot overlap**.

</div>
<div class="card blue">

### 🛤️ Async copies in a stream
Pass `stream=s`. A plain `to_device` **waits**.

</div>
<div class="card orange">

### 🧩 Independent chunks
Chunk *k*'s result must **not be needed** by chunk *k+1*.

</div>
</div>

Also: the kernel needs **enough work** (a kernel that is 1 % of the time leaves nothing to hide). Pinned memory is scarce: **pin what you stream**, not everything.

---

## 🧑‍💻 Code: one stream per chunk

```python
h_in  = cuda.pinned_array(n, dtype=np.float32)   # page-locked
h_out = cuda.pinned_array(n, dtype=np.float32)
streams = [cuda.stream() for _ in range(K)]
chunk = n // K

for k, s in enumerate(streams):
    lo, hi = k * chunk, (k + 1) * chunk
    d_in  = cuda.to_device(h_in[lo:hi], stream=s)
    d_out = cuda.device_array(chunk, dtype=np.float32, stream=s)
    heavy[blocks, TPB, s](d_in, d_out, iters)
    d_out.copy_to_host(h_out[lo:hi], stream=s)
cuda.synchronize()                               # all streams
```

📁 `week09/10_streams_overlap.py`

---

<style scoped>section{font-size:23px} pre{font-size:15px;line-height:1.15}</style>

## 🏭 Walkthrough: how 3 streams overlap on hardware

The Python `for` loop queues commands **instantly** (asynchronously).
The GPU routes them to **three independent hardware engines**:

```text
Time →        t1           t2           t3           t4           t5
          ┌─────────┐  ┌─────────┐  ┌─────────┐
H2D Copy  │ Chunk 0 │  │ Chunk 1 │  │ Chunk 2 │
          └─────────┘  └─────────┘  └─────────┘
                       ┌─────────┐  ┌─────────┐  ┌─────────┐
SM Compute             │ Chunk 0 │  │ Chunk 1 │  │ Chunk 2 │
                       └─────────┘  └─────────┘  └─────────┘
                                    ┌─────────┐  ┌─────────┐  ┌─────────┐
D2H Copy                            │ Chunk 0 │  │ Chunk 1 │  │ Chunk 2 │
                                    └─────────┘  └─────────┘  └─────────┘
```

- **Inside one stream (e.g., Chunk 0):** `H2D → Compute → D2H` runs strictly in order.
- **Across streams at `t3`:** All three hardware engines work in parallel (`D2H 0`, `Compute 1`, `H2D 2`).
- **Without pinned memory:** The `H2D` engine stalls waiting on the CPU, collapsing the staircase back into serial execution.

---

## ▶️ Run it

```bash
python 10_streams_overlap.py                  # serial vs 4 streams, verified against NumPy
python 10_streams_overlap.py --chunks 8       # more chunks: more overlap, more launch overhead
python 10_streams_overlap.py --iters 1        # a light kernel: nothing to hide
```

- With K balanced stages the ideal speedup is **3K / (K + 2)**: 2× for K = 4. The longest stage sets the limit.
- Predict first: what happens at `--iters 1`? At `--chunks 1`?
- A timeline (`nsys`, see *Profiling*) shows copies and kernels side by side.
- **Returns as:** `pin_memory=True` + `non_blocking=True` (Week 10); DDP overlap (Weeks 11–12).

---

<!-- _class: lead -->

# 🧪 Lab and wrap-up

---

<!-- _footer: "Lab notebook: week09/lab_week09_memory_hierarchy.ipynb" -->

## 🧪 Lab: six experiments, one story

| # | task | file |
|---|---|---|
| 1 | thread-per-row vs thread-per-column sums, 4096² — achieved GB/s | `02_coalescing.py` |
| 2 | naive → tiled, T ∈ {8, 16, 32}, N ∈ {512, 1024, 2048} | `03`, `04`, `05` |
| 3 | remove each barrier, 3 runs each | `04 --race` |
| 4 | transpose: padded vs unpadded | `06_bank_conflicts.py` |
| 5 | block reduce → shared softmax; profile or measure | `07`, `08`, `09` |
| 6 | serial vs streams; chunks and kernel length | `10_streams_overlap.py` |

**Deliverable:** the results tables in the notebook, plus one paragraph per experiment.

---

## 🧾 Answer sheet

1. **Bytes:** how many bytes did your kernel move from VRAM? Show the arithmetic.
2. **Intensity:** FLOPs ÷ bytes. Where does it sit on the roofline?
3. **What changed:** did tiling reduce *traffic*, raise *occupancy*, or both?
4. **Correctness:** which barrier can you remove safely? Justify it with your three runs from experiment 3.

Report the sentence: *"my kernel is ______-bound, because ______ GB/s
out of 320, and ______ GFLOP/s out of 8,100."*

---

## 🧱 Further notes: matrices that are not a multiple of T

What if the dimensions $(M \times K)$ and $(K \times N)$ are **not** multiples of T? Two rules prevent deadlocks and garbage math:

<div class="cols">
<div class="card green">

### 0️⃣ Pad loads with `0.0`
Pad out-of-bounds loads with **`0.0`**. Never leave shared memory uninitialized.

</div>
<div class="card blue">

### 🚧 Barriers stay outside `if`
Keep `cuda.syncthreads()` **outside any `if` block**.

</div>
</div>

---

## 🧱 Safe boundary checks in the tiled loop

```python
        for k0 in range(0, K, T):
            # 1. Load tile of A (or pad with 0.0 if outside matrix)
            if row < M and (k0 + tx) < K:
                sA[ty, tx] = A[row, k0 + tx]
            else:
                sA[ty, tx] = 0.0

            # 2. Load tile of B (or pad with 0.0 if outside matrix)
            if (k0 + ty) < K and col < N:
                sB[ty, tx] = B[k0 + ty, col]
            else:
                sB[ty, tx] = 0.0

            cuda.syncthreads()              # ALL threads reach this

            for k in range(T):
                acc += sA[ty, k] * sB[k, tx]
            cuda.syncthreads()              # ALL threads reach this

        # 3. Guard only the final write to global memory
        if row < M and col < N:
            C[row, col] = acc
```

---

## 🔭 Forward map

| today | where it reappears |
|---|---|
| shared-memory tiling | FlashAttention-style tiling of attention (reading list) |
| coalescing | tensor layout, `.contiguous()`, channels-last in PyTorch |
| block reduction | softmax / layernorm kernels inside PyTorch |
| occupancy | batch-size and kernel tuning (Weeks 10–13) |
| achieved GB/s | LLM inference: tokens/s tracks memory bandwidth (Week 13) |
| profiler timelines | GPU idle time → DataLoader (Week 10) |
| streams, pinned memory | `non_blocking` copies and `pin_memory` (Week 10); DDP overlap (Weeks 11–12) |

---

## ✅ Takeaways: correctness first

1. **Shared memory is a user-managed cache.** You decide what to keep.
2. **Two barriers per tile**, both reached by every thread in the block.
3. A race gives a *nondeterministic* answer, not a crash. Verify against NumPy.
4. `cuda.shared.array` needs a constant shape → factory to sweep tile sizes.
5. Reductions: `tid < stride`, barrier outside the `if`, pad with the identity.

---

## ⚡ Takeaways: then performance

1. **Count bytes before you optimise.** Intensity tells you which ceiling.
2. **Coalescing:** consecutive lanes → consecutive addresses (in a row-major matrix, one thread per *column*). 8× at stake.
3. **Tiling** divides global traffic by T and multiplies intensity by T.
4. **Bank conflicts** are about *stride*, not about using shared memory.
   Same address across a warp is a broadcast, not a conflict.
5. **Occupancy is a proxy**, not a goal. Sweep and measure.
6. Time with CUDA events: warm up, then take the median of ≥ 5 runs.
7. **Streams** overlap copies with compute, but only with pinned memory, async copies and independent chunks.

---

## ✍️ Before next week

- **Finish the lab** (`lab_week09_memory_hierarchy.ipynb`) with your results tables and one paragraph per experiment.
- **Re-run** `04_matmul_tiled.py --race` a few more times. How often does a broken kernel look correct?
- **Read:** NVIDIA CUDA C++ Best Practices Guide, *Memory Optimizations*.
- **Think ahead:** a profiler timeline shows gaps between kernels. What could make a GPU wait for its *input*? Streams hide copies; Week 10 is about the CPU side of the same problem.

---

## 📚 References

- NVIDIA, *CUDA C++ Programming Guide* and *Best Practices Guide*: memory hierarchy, coalescing, shared memory, bank conflicts, effective bandwidth.
- Mark Harris, *Using Shared Memory in CUDA C/C++* and *An Efficient Matrix Transpose in CUDA C/C++*, NVIDIA Technical Blog, 2013.
- V. Volkov, *Better Performance at Lower Occupancy*, GTC 2010.
- [Numba CUDA: Memory](https://nvidia.github.io/numba-cuda/user/memory.html) · [Writing Kernels](https://nvidia.github.io/numba-cuda/user/kernels.html) · [Course repository](https://github.com/adaskin/hpc-for-ai-course)

Figures `img/w09-*.svg` are drawn by `week09/make_figures.py`; the `img/w09-*.png` figures are from NVIDIA's CUDA Programming Guide (links on the slides). See `img/CREDITS.md`.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank You

### Next week: feeding the GPU — DataLoader pipelines (Week 10)

*YZM 513 – İstanbul Medeniyet Üniversitesi – Fall 2026*
