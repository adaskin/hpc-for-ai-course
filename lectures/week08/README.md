## Week 8: your first CUDA kernels

Slides: `../08cuda-kernel-numba-cuda.md` (Marp). Part III interleaves seven CUDA C/C++ worked examples with the concept slides; each has a **problem**, a **step-by-step** walkthrough with worked index tables, the **full code** and the **expected output**.

## CUDA C/C++ examples (`cuda_c/`, Part III of the deck)

Each file is a complete, standalone program. Build and run with `nvcc -o NAME NAME.cu && ./NAME`.

| File | Concept | Deterministic output? |
|---|---|---|
| `01_hello_indices.cu` | The core formula `blockIdx.x * blockDim.x + threadIdx.x`, 3 blocks × 4 threads | Same 12 lines; **order may vary** |
| `02_vector_add_small.cu` | Full workflow on `N = 10`: `malloc` → `cudaMalloc` → `cudaMemcpy` → launch → `cudaMemcpy` → free | Yes |
| `03_bounds_check.cu` | What a missing `if (i < n)` does (guard cells make the damage visible and safe) | Yes |
| `04_launch_config.cu` | `blocks = ceil(N / threads)`, device limits, floor-division mistake | Yes, except Part B (your GPU's name and limits) |
| `05_error_checking.cu` | `CUDA_CHECK`, `cudaGetLastError`, asynchronous kernel errors, sticky errors | Yes; message wording can vary by CUDA version |
| `06_matrix_2d.cu` | 2D indexing on a 5 × 7 matrix, row-major flat index | Yes |
| `07_grid_stride.cu` | Grid-stride loop: who handles which element, SAXPY for three values of `N` | Yes |

## Numba CUDA

| File | What it shows |
|---|---|
| `06_indexing_numba.py` | Examples 1, 2, 3, 6 and 7 of the CUDA C set, in Numba CUDA |
| `01_vector_add.py` | Vector add, 10M elements, CUDA-event timing and bandwidth |
| `02_relu_kernel.py` | Element-wise ReLU on AI-shaped data |
| `03_fused_relu_bias.py` | Fused vs unfused ReLU + bias |
| `04_softmax_kernel.py` | Deliberately naïve row-wise softmax (why reductions need cooperation) |
| `05_benchmark_kernels.py` | Benchmark: your kernel vs CuPy vs NumPy |

## Labs

- `lab_week08_cuda_c.ipynb`: **warm-up**. Compile and run the seven CUDA C programs on Colab, predict outputs, change one thing per example.
- `lab_week08_kernels.ipynb`: **main lab**. Four Numba CUDA exercises (vector add, ReLU, fusion, vs CuPy) and a results table.

## Setup

Colab: Runtime → Change runtime type → **T4 GPU**. `nvcc` is already there for the C examples; the Numba notebook installs `numba-cuda`.

Local machine with an NVIDIA GPU and the CUDA toolkit:

```bash
cd cuda_c && nvcc -o 01_hello_indices 01_hello_indices.cu && ./01_hello_indices
pip install "numba-cuda[cu12]"     # match your CUDA version
python 06_indexing_numba.py
```

No GPU? `NUMBA_ENABLE_CUDASIM=1 python 06_indexing_numba.py` runs the Numba kernels in a (slow) CPU simulator, enough to check the index arithmetic (the order of `print` lines is not meaningful there).

## Expected take-aways

1. Every thread computes its own index, `i = blockIdx.x * blockDim.x + threadIdx.x`, and then does a small, well-defined piece of work.
2. `blocks = (N + threads - 1) / threads` launches at least `N` threads; the extra threads must be stopped by `if (i < n)`.
3. The CPU and the GPU have separate memories: allocate, copy in, launch, copy out, free.
4. Check every CUDA call, and call `cudaGetLastError()` after every launch; kernel crashes surface at the next synchronising call.
5. 2D launches give `(col, row)` from `(x, y)`; a grid-stride loop lets one fixed launch handle any `N`.
