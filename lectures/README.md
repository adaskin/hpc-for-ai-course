# Week 6 — GPU Architecture & First Steps with CuPy

Companion code for `lectures/06gpu-architecture-and-cupy.md`.

The **lab** is `lab_week06_cupy.ipynb` (Colab-ready): open it in Colab,
enable a GPU runtime (Runtime → Change runtime type → T4), Runtime → Run all.
Students fill in the 📝 results tables and answer the wrap-up questions.

## Scripts (class demos, same content as the notebook cells)

| File | What it shows |
|---|---|
| `01_gpu_info.py` | `nvidia-smi` + device properties via the CuPy runtime API |
| `02_cupy_matmul_benchmark.py` | matmul sweep n=256…8192: NumPy vs CuPy compute-only vs end-to-end |
| `03_cupy_elementwise.py` | ReLU / sigmoid on 10M floats — the pure SIMT workload |
| `04_transfer_overhead.py` | upload bandwidth vs size + the "transfer trap" on tiny arrays |
| `05_cupy_vs_numpy_svd.py` | SVD: LAPACK vs cuSOLVER (Project 1 callback) |
| `06_crossover_finder.py` | doubling search for the n where GPU end-to-end beats CPU |

## Setup

Local machine with an NVIDIA GPU:

```bash
pip install cupy-cuda12x   # match your CUDA version, see https://docs.cupy.dev/en/stable/install.html
python 01_gpu_info.py
```

Google Colab: `!pip install -q cupy-cuda12x` in the first cell (the notebook does this).

## Expected take-aways

1. GPU wins only when compute time ≫ PCIe transfer time (n ≥ ~1000–2000 for matmul on T4).
2. Element-wise ops on large arrays: 5–30× on-device speedup, memory-bandwidth bound.
3. The four course questions on the GPU: split = one thread per element,
   assign = hardware block scheduler, move = `asarray`/`.get()` over PCIe, wait = async streams.
