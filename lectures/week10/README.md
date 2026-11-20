## Week 10: feeding the GPU (data pipelines for deep learning)

Slides: `../10data-pipelines.md` (Marp). The deck follows PyTorch's official tutorial *Data Loading Optimization in PyTorch*: start from the slowest loader and add one change at a time (workers, pinned memory, persistent workers, side-stream copy, `__getitems__`), measuring each step. It reviews Week 9 in one slide, explains the PyTorch basics for newcomers in one slide, and adds the measuring tools (hidden syncs, timing recipe, `torch.profiler`).

**Rendering the slides.** The deck is plain Markdown plus Marp directives; no HTML is needed.

## Scripts

Run them from this directory. Every configuration is checked to see the same data as the baseline before it is timed.

| File | Deck topic | What it shows | GPU? |
|---|---|---|---|
| `01_cost_model.py` | idle GPU, worker sizing | serial vs overlapped step time, workers needed for a given GPU step | no |
| `02_ladder.py` | the ladder | baseline, workers, `pin_memory`, persistent workers, side-stream copy, `__getitems__`; speedup per step and in total | runs on CPU; `pin_memory` and side-stream steps are skipped without a GPU |
| `03_pinned_transfer.py` | moving bytes safely | pageable vs pinned vs "pin by hand" copy times; the two `non_blocking` hazards with their fix | **yes** |
| `04_hidden_syncs.py` | hidden syncs | `loss.item()` every step vs every N steps; data wait vs step time | runs on CPU, but the effect only shows on a GPU |
| `05_profile_pipeline.py` | profiler | `torch.profiler` with `data_wait` labels, exported for Perfetto | runs on CPU |
| `pipeline_utils.py` | all | synthetic dataset, `__getitems__` dataset, prefetcher, timed training loop | |
| `make_figures.py` | all | regenerates the five `../img/w10-*.svg` figures (`python make_figures.py`; `--png DIR` for previews) | |

Most scripts take size options; run with `--help`.

## Lab

Run `02_ladder.py` and fill in the table (time, speedup vs previous step, cumulative speedup, data-wait share), then trace the baseline and the best loader with `05_profile_pipeline.py`. `03` and `04` need a GPU (Colab T4). The answer sheet is on the last slides. A Colab notebook can wrap these scripts, as in Week 9.

## Setup

```bash
pip install torch numpy matplotlib        # matplotlib only for make_figures.py
python 02_ladder.py
```

Colab: Runtime, Change runtime type, **T4 GPU**. Record `torch.__version__`, `os.cpu_count()` and the Python version with your results: the DataLoader's start method depends on the Python version.

## Expected take-aways

1. Measure the wait before you touch any knob; the knobs only help an input-bound job.
2. Climb the ladder in order and re-measure. Some steps (`pin_memory`, the prefetcher) may give nothing.
3. The biggest lever is usually less work per sample (batched calls, caching, smaller dtypes), not more workers.
4. Copy/compute overlap needs a pinned source, a separate stream and a free copy engine; `non_blocking` without a sync is a race.
5. Hidden syncs (`.item()` and similar) can make an idle GPU look like a slow loader. Use the trace.

## Preparation notes: what was checked, and what was not

The deck was prepared without a GPU. Scripts `01`, `02`, `04` and `05` were run on a CPU machine (Python 3.13, PyTorch 2.15); `03` and the GPU-only steps of `02` and `04` were **not run** and need a first run on a T4 before class.

| Claim in the deck | Status |
|---|---|
| The official tutorial's ladder and its reported speedups (about 3.4x workers, 1.0x `pin_memory`, 1.1x persistent workers, 1.0x prefetcher, about 4x `__getitems__`) | read from the tutorial page (part of it as a summary); its hardware is not stated; re-check against the page and re-run on the T4 before quoting |
| Overlap needs a free DMA engine, a separate non-default stream and a pinned source; `pin_memory().to(..., non_blocking=True)` by hand up to 2x slower; the two hazards | the official `non_blocking` / `pin_memory()` guide (hazards and numbers read directly, the three requirements from a summary) |
| `in_order` and its warning; `forkserver` as the default on Python 3.14+; `prefetch_factor` default 2 per worker; `persistent_workers`; the `__getitems__` hook | PyTorch source docstrings on `main`; check them against the PyTorch version on Colab |
| Hidden-sync list, `set_to_none`, `cudnn.benchmark` | official Performance Tuning Guide (part of it as a summary); re-read before teaching |
| Random seeds differ per worker; Python-object datasets grow memory in workers | general guidance; the DataLoader docs' randomness notes were **not** read: check them before teaching this slide |
| Profiler: `schedule`, `record_function`, `export_chrome_trace`; viewer | official Profiler recipe; the TensorBoard plugin was not mentioned on the page I read, so the deck uses Perfetto / Chrome trace |
| `tf.data` code (`map` with `num_parallel_calls`, `cache`, `prefetch`, `AUTOTUNE`) and the toy timings (0.517 s sequential vs 0.395 s parallel map) | read from the official `tf.data` performance guide page; the deck's three-call chain is our own combination of the page's separate snippets |
| Grain `MapDataset` chain, `ReadOptions(num_threads=16, prefetch_buffer_size=500)`; Hugging Face `with_format("torch")` + `DataLoader(num_workers=4)` | read from the Grain basic tutorial and the Hugging Face `use_with_pytorch` page |
| JAX `ds.shard(... process_index())` example | read from the JAX distributed data loading page as a **summary**, not verbatim: re-check before teaching |
| NVIDIA DALI example, "operators can run on the GPU", "plugs into PyTorch, TensorFlow and JAX" | the code is from a summary of DALI's *Pipeline mode* page; the GPU and framework claims are general knowledge, **not** read from the docs: check before teaching |
| `w10-four-cases.svg` | our schematic of the four profiler traces the official `non_blocking` guide describes; we could not download the guide's own images |
| Illustrative numbers (30 ms data + 50 ms GPU; 5 ms per call) | schematic: replace with the numbers from your own run |

The 5 ms "fixed cost per call" in the synthetic dataset stands for per-call overhead (opening a file, calling a decoder). `__getitems__` helps because it pays that cost once per batch; real datasets benefit only if they can read or decode a batch in one go.
