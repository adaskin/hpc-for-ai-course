"""
Week 3 – Four Patterns for Sharing Data with Worker Processes
================================================================

Demonstrates, with timing:
  0. Naive pickling          – the "obvious" slow way (anti-pattern)
  1. Indices + fork          – zero-copy COW reads (Linux only)
  2. shared_memory           – ANONYMOUS mmap: shared RAM segment (cross-platform)
  3. Filenames               – workers load their own slice from DISK
  4. np.memmap / mmap_mode   – FILE-BACKED mmap: single process, data > RAM

KEY DISTINCTION (do not confuse):
  • shared_memory (Pattern 2) uses ANONYMOUS mmap.
    → Creates a shared RAM segment for IPC. No file on disk.
    → Answers: "How do processes share data already in RAM?"

  • np.memmap / mmap_mode="r" (Patterns 3 & 4) uses FILE-BACKED mmap.
    → Maps a file into virtual address space. OS pages in on demand.
    → Answers: "How do I work with data too large for RAM?"

  Same OS syscall (mmap). Completely different problems.

Usage:
    python four_patterns.py

Notes:
  • Worker functions MUST be at module level (picklable for spawn).
  • Pattern 1 requires fork; on macOS/Windows it will be skipped.
  • Results are verified against a serial NumPy reference.
"""

import os
import sys
import time
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np

# ───────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────
SIZE      = 20_000_000        # elements (reduce for slower machines)
DTYPE     = np.float64
N_WORKERS = 4                 # keep 4 for typical laptops
SEED      = 513

# ───────────────────────────────────────────────────────────────────
# The compute task – identical across all patterns
# ───────────────────────────────────────────────────────────────────
def transform(x):
    """~15–20 FLOPs per element. Compute-heavy, NOT memory-bandwidth-bound.

    This matters: if the task were bandwidth-bound (e.g. just np.sum),
    splitting across cores would give no speedup even with zero IPC,
    because all cores would fight for the same DRAM bandwidth.
    """
    return np.sin(x) * np.cos(x) + x ** 2 - np.sqrt(np.abs(x) + 1e-9)

# ───────────────────────────────────────────────────────────────────
# Module-level global for Pattern 1 (fork + copy-on-write)
# ───────────────────────────────────────────────────────────────────
data = None   # populated in main BEFORE fork-based workers are created


# ═══════════════════════════════════════════════════════════════════
# Pattern 0 – Naive pickling (the anti-pattern)
# ═══════════════════════════════════════════════════════════════════
def sum_chunk_pickled(chunk):
    """Receives the whole chunk as an argument → pickled over IPC."""
    return float(np.sum(transform(chunk)))


def run_pattern_0_naive_pickle(arr, n_workers):
    """Pass actual array chunks to workers. Each chunk is pickled."""
    chunks = np.array_split(arr, n_workers)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        partials = list(ex.map(sum_chunk_pickled, chunks))
    elapsed = time.perf_counter() - t0
    return sum(partials), elapsed, None


# ═══════════════════════════════════════════════════════════════════
# Pattern 1 – Indices + fork (zero copy on Linux)
# ═══════════════════════════════════════════════════════════════════
def sum_bounds(bounds):
    """Reads data[s:e] from inherited parent memory (COW).

    Only `bounds` (~16 bytes) crosses the IPC boundary.
    The array itself is never copied or pickled.
    Under the hood: fork() duplicates page tables; pages are
    marked read-only; reads hit the same physical frames.
    """
    s, e = bounds
    return float(np.sum(transform(data[s:e])))


def run_pattern_1_indices(arr, n_workers):
    global data
    data = arr   # set BEFORE the pool forks so children inherit it

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return None, None, "fork not available on this platform"

    bounds = [(i * SIZE // n_workers, (i + 1) * SIZE // n_workers)
              for i in range(n_workers)]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        partials = list(ex.map(sum_bounds, bounds))
    elapsed = time.perf_counter() - t0
    return sum(partials), elapsed, None


# ═══════════════════════════════════════════════════════════════════
# Pattern 2 – shared_memory (cross-platform)
#
# Uses ANONYMOUS mmap internally:
#   • OS allocates a shared RAM segment (no file on disk).
#   • Parent writes data into it ONCE.
#   • Workers attach by name and read the same physical pages.
#   • This is IPC (inter-process communication), NOT disk I/O.
# ═══════════════════════════════════════════════════════════════════
def sum_shm(args):
    """Attaches to the shared RAM segment by NAME.

    Only the name string (~50 bytes) is pickled.
    The data is accessed via the shared mapping — no IPC copy.
    """
    name, s, e = args
    seg = shared_memory.SharedMemory(name=name)
    try:
        arr = np.ndarray((SIZE,), dtype=DTYPE, buffer=seg.buf)
        return float(np.sum(transform(arr[s:e])))
    finally:
        seg.close()   # close this process's view (does NOT free the segment)


def run_pattern_2_shared_memory(arr, n_workers):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    try:
        # One upfront copy into the shared RAM segment
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shm_arr[:] = arr

        bounds = [(shm.name,
                   i * SIZE // n_workers,
                   (i + 1) * SIZE // n_workers)
                  for i in range(n_workers)]

        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            partials = list(ex.map(sum_shm, bounds))
        elapsed = time.perf_counter() - t0
        return sum(partials), elapsed, None
    finally:
        shm.close()
        shm.unlink()   # free the OS segment. Forget this → leak.


# ═══════════════════════════════════════════════════════════════════
# Pattern 3 – Filenames (cross-platform, disk-based)
#
# Each worker loads its own chunk from disk.
# Uses FILE-BACKED mmap (mmap_mode="r"):
#   • The .npy file is mapped into virtual address space.
#   • OS pages in only the pages the worker actually touches.
#   • No IPC. No shared RAM. Workers are fully independent.
# ═══════════════════════════════════════════════════════════════════
def load_and_sum(path):
    """Worker loads its own slice from disk via file-backed mmap."""
    arr = np.load(path, mmap_mode="r")   # does NOT load entire file into RAM
    return float(np.sum(transform(arr)))


def write_chunks_to_disk(arr, n_workers, out_dir):
    paths = []
    chunk = SIZE // n_workers
    for i in range(n_workers):
        s = i * chunk
        e = (i + 1) * chunk if i < n_workers - 1 else SIZE
        path = os.path.join(out_dir, f"chunk_{i:02d}.npy")
        np.save(path, arr[s:e])
        paths.append(path)
    return paths


def run_pattern_3_filenames(arr, n_workers, tmp_dir):
    paths = write_chunks_to_disk(arr, n_workers, tmp_dir)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        partials = list(ex.map(load_and_sum, paths))
    elapsed = time.perf_counter() - t0
    return sum(partials), elapsed, None


# ═══════════════════════════════════════════════════════════════════
# Pattern 4 – np.memmap / mmap_mode (single process, disk-based)
#
# FILE-BACKED mmap, same mechanism as Pattern 3 but:
#   • No worker pool. Single process.
#   • The point is NOT parallelism.
#   • The point is: work with data LARGER than RAM.
#   • OS pages in only what you touch.
#
# This is what HuggingFace `datasets`, large embedding tables,
# and memory-mapped training corpora use under the hood.
# ═══════════════════════════════════════════════════════════════════
def run_pattern_4_memmap(npy_path):
    """Single-process memory-mapped read. No workers, no IPC.

    np.load(path, mmap_mode='r') understands the .npy header and
    maps only the data payload. The OS pages in only what's touched.
    """
    t0 = time.perf_counter()
    mm = np.load(npy_path, mmap_mode="r")
    result = float(np.sum(transform(mm)))
    elapsed = time.perf_counter() - t0
    return result, elapsed, None


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    arr = rng.random(SIZE, dtype=DTYPE)

    print("=" * 70)
    print(" Week 3 – Data-Sharing Patterns for Worker Processes")
    print("=" * 70)
    print(f" Data:      {SIZE:,} elements "
          f"({DTYPE.__name__}, {arr.nbytes / 1e6:.1f} MB)")
    print(f" Workers:   {N_WORKERS}")
    print()

    # ── Serial baseline (also the correctness reference) ─────────────
    t0 = time.perf_counter()
    reference = float(np.sum(transform(arr)))
    t1 = time.perf_counter()
    serial_time = t1 - t0

    # ── Run every pattern ────────────────────────────────────────────
    rows = []   # (label, time, value, verified, note)
    rows.append(("Serial (baseline)", serial_time, reference, True, ""))

    s, t, note = run_pattern_0_naive_pickle(arr, N_WORKERS)
    rows.append(("0. Naive pickle", t, s,
                 np.isclose(s, reference), note or ""))

    s, t, note = run_pattern_1_indices(arr, N_WORKERS)
    if s is not None:
        rows.append(("1. Indices + fork", t, s,
                     np.isclose(s, reference), note or ""))
    else:
        rows.append(("1. Indices + fork", None, None, None, note))

    s, t, note = run_pattern_2_shared_memory(arr, N_WORKERS)
    rows.append(("2. shared_memory", t, s,
                 np.isclose(s, reference), note or ""))

    with tempfile.TemporaryDirectory() as tmp:
        s, t, note = run_pattern_3_filenames(arr, N_WORKERS, tmp)
        rows.append(("3. Filenames (disk)", t, s,
                     np.isclose(s, reference), note or ""))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "full.npy")
        np.save(path, arr)
        s, t, note = run_pattern_4_memmap(path)
        rows.append(("4. np.memmap (1 proc)", t, s,
                     np.isclose(s, reference), note or ""))

    # ── Print results table ──────────────────────────────────────────
    print()
    print("-" * 70)
    print(f" {'Pattern':<26} {'Time (s)':>9}   {'Speedup':>7}   Verified")
    print("-" * 70)
    for label, t, val, ok, note in rows:
        if t is None:
            print(f" {label:<26} {'—':>9}   {'—':>7}   skipped ({note})")
            continue
        speedup = serial_time / t if t > 0 else 0.0
        flag = "✓" if ok else "✗ WRONG"
        print(f" {label:<26} {t:>9.3f}   {speedup:>6.2f}×   {flag}")
    print("-" * 70)
    print()
    print(f" Reference value:  {reference:.6f}")
    print()

    # ── Interpretation ───────────────────────────────────────────────
    print(" Interpretation:")
    print("   0. Naive pickle: pays full pickling cost. Workers starve.")
    print("   1. Indices+fork: cheapest IPC. Only (start,end) crosses boundary.")
    print("      Uses COW (copy-on-write) on inherited pages. Linux only.")
    print("   2. shared_memory: ANONYMOUS mmap. Shared RAM segment for IPC.")
    print("      One upfront copy, then zero per worker. Cross-platform.")
    print("   3. Filenames: FILE-BACKED mmap. Workers read from disk.")
    print("      No shared RAM. No IPC. Scales to huge datasets.")
    print("   4. np.memmap: FILE-BACKED mmap. Single process. No parallelism.")
    print("      Purpose: work with data LARGER than RAM.")
    print()
    print(" Key distinction:")
    print("   shared_memory  = anonymous mmap  = RAM-based IPC")
    print("   np.memmap      = file-backed mmap = disk-based, pages on demand")
    print("   Same OS mechanism (mmap). Different problems.")
    print()
    print(" Profile the pickling cost explicitly:")
    print("   python -m cProfile -s cumtime four_patterns.py")