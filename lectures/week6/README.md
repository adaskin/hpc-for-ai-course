## The main lab: one MLP, NumPy → CuPy → PyTorch

**`lab_week06_train_mlp_gpu.ipynb`** (Colab-ready): Runtime → Change runtime type → **T4 GPU**, then Run all (about 5–8 min).
It trains one 784-512-10 MLP on MNIST, written as matrices, and carries it through the course so far; it is the running example of the Week 1–6 review (deck Part VII).


`lab_week06_cupy.ipynb` is the benchmark-only lab (matmul sweep, ReLU, transfer trap, SVD). It overlaps with Part C;

## Scripts (class demos for the benchmark lab)

| File | What it shows |
|---|---|
| `01_gpu_info.py` | `nvidia-smi` + device properties via the CuPy runtime API |
| `02_cupy_matmul_benchmark.py` | matmul sweep n=256…8192: NumPy vs CuPy compute-only vs end-to-end |
| `03_cupy_elementwise.py` | ReLU / sigmoid on 10M floats, the pure SIMT workload |
| `04_transfer_overhead.py` | upload bandwidth vs size + the "transfer trap" on tiny arrays |
| `05_cupy_vs_numpy_svd.py` | SVD: LAPACK vs cuSOLVER (Project 1 callback) |
| `06_crossover_finder.py` | doubling search for the n where GPU end-to-end beats CPU |

## Setup

Colab: nothing to install (the notebook installs the matching `cupy-cuda1Nx`). Local machine with an NVIDIA GPU:

```bash
pip install cupy-cuda12x   # match your CUDA version, see https://docs.cupy.dev/en/stable/install.html
python 01_gpu_info.py
```

## Expected take-aways

1. A training step is a few matmuls plus cheap element-wise work; the GPU wins when compute per kernel ≫ launch and transfer overhead.
2. Time GPU code only after `synchronize()`; keep data on the device; stay in `float32`.
3. Matmul is compute-bound (FLOP/byte grows like n/6), element-wise ops are bandwidth-bound.
4. The four course questions on the GPU: split = one thread per element, assign = hardware block scheduler, move = `asarray`/`asnumpy` over PCIe, wait = asynchronous launches + `synchronize`.
