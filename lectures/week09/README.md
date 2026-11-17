## Week 9: the CUDA memory hierarchy and profiling

Slides: `../09memory-hierarchy-and-profiling.md` (Marp). The deck follows one idea: a kernel is fast when **each byte fetched from VRAM is used many times**. Parts I–IV (counting bytes, the memory hierarchy, coalescing, shared-memory tiling) are the spine; Parts V–VIII (reductions, occupancy, profiling, streams) can move to self-study if time is short.

## Scripts

Each script is complete and prints its own results; every kernel is verified against NumPy or SciPy before it is timed. Run them from this directory.

| File | Deck part | What it shows |
|---|---|---|
| `01_traffic_accounting.py` | I | The traffic model for naive and tiled matmul against the T4 ceilings. **No GPU needed.** |
| `02_coalescing.py` | III | Sum of a 4096² matrix with one thread per row (strided) vs one thread per column (coalesced); achieved GB/s |
| `03_matmul_naive.py` | I | Naive matmul, timed and compared with the no-reuse traffic model |
| `04_matmul_tiled.py` | IV | Shared-memory tiled matmul; `--race` removes each `syncthreads()` in turn to show a data race |
| `05_tile_sweep.py` | IV, VI | Naive vs tiled (T = 8, 16, 32) vs cuBLAS, with theoretical occupancy |
| `06_bank_conflicts.py` | IV | Transpose with and without a padded shared tile |
| `07_block_reduce.py` | V | Shared-memory tree reduction with one atomic per block vs one atomic per thread |
| `08_softmax_shared.py` | V | Week 8's softmax redone with a cooperative reduction (one block per row) |
| `09_profile_and_measure.py` | VII | Tries Nsight Compute, falls back to events + achieved GB/s and GFLOP/s; `--torch` adds `torch.profiler` |
| `10_streams_overlap.py` | VIII | Serial upload → kernel → download vs K chunks in K streams with pinned memory (`cuda.stream`) |
| `gpu_utils.py` | all | `time_ms` (CUDA-event timing), `gbs`, `gflops`, `device_report`, `kernel_resources`, `theoretical_occupancy` |
| `make_figures.py` | all | Regenerates the ten `../img/w09-*.svg` figures (`python make_figures.py`; `--png DIR` for previews) |

Most scripts take a size option (`--n`, `--sizes`, `--rows`, `--log2n`); run with `--help`.

## Lab

`lab_week09_memory_hierarchy.ipynb` (Colab-ready, T4): six experiments, each with a prediction, a results table and questions. It runs the scripts above, so it needs no code of its own.

## Setup

Colab: Runtime → Change runtime type → **T4 GPU**. The notebook installs `numba-cuda` and CuPy.

Local machine with an NVIDIA GPU:

```bash
pip install "numba-cuda[cu12]" cupy-cuda12x scipy     # match your CUDA version
python 01_traffic_accounting.py
python 04_matmul_tiled.py --race
```

No GPU? `NUMBA_ENABLE_CUDASIM=1 python 04_matmul_tiled.py --n 16 --tile 8` runs the kernels in the (very slow) CPU simulator. That is enough to check results, including that a missing barrier gives a wrong answer, but the timings it prints are meaningless; keep the sizes tiny.

Ceilings used throughout are for a **T4**: 320 GB/s memory bandwidth and 8,100 GFLOP/s FP32 (ridge point ≈ 25 FLOP/byte). On another GPU, pass `--peak` to `02_coalescing.py` and read the limits from `device_report()`.

## Expected take-aways

1. **Count bytes before optimising.** Arithmetic intensity (FLOP per byte from VRAM) tells you which ceiling you are under.
2. **Coalescing:** consecutive lanes should read consecutive addresses. In a row-major matrix, one thread per *column* does.
3. **Tiling** divides global-memory traffic by the tile size T and multiplies arithmetic intensity by T. Shared memory is a user-managed cache.
4. **Two barriers per tile**, reached by every thread of the block. A missing barrier gives a nondeterministic answer, not a crash.
5. **Bank conflicts** come from stride, not from using shared memory; a broadcast (same address) is not a conflict. Padding a transpose tile to `(T, T+1)` removes the 32-way conflict.
6. **Reductions** need cooperation: `tid < stride`, the barrier outside the `if`, padding with the identity, and one atomic per block, not per thread.
7. **Occupancy is a proxy, not a goal.** Sweep and measure. Time with CUDA events, warm up, and report a median.
8. **Streams** overlap copies with compute, but only with pinned host memory, asynchronous copies and independent chunks. The ideal speedup for K balanced stages is 3K / (K + 2).
