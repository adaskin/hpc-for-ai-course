"""
Benchmark & Plotting Utilities for Project 1

Use these across all phases. They help you:
- Time your code reliably (with warmup)
- Measure memory usage
- Verify correctness
- Generate publication-quality plots
- Save raw data for your report
"""

import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import csv


# ========================================================================
# TIMING & MEMORY
# ========================================================================

def benchmark(func, *args, warmup=3, runs=5, **kwargs):
    """
    Benchmark a function with warmup and multiple runs.

    Critical for Numba: the first call compiles the function.
    Always use warmup >= 3 for JIT-compiled code.

    Returns:
        dict with mean, std, min, max, and the last result
    """
    # Warmup runs (results discarded)
    for _ in range(warmup):
        result = func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "results": result,
        "raw_times": times.tolist(),
    }


def benchmark_with_memory(func, *args, **kwargs):
    """
    Benchmark a single run with peak memory measurement.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "time": t1 - t0,
        "peak_memory_mb": peak / (1024 * 1024),
        "results": result,
    }


# ========================================================================
# CORRECTNESS
# ========================================================================

def verify_results(result_a, result_b, tol=1e-5):
    """
    Verify two top-k result lists are equivalent.
    Checks both indices and scores within tolerance.
    """
    if len(result_a) != len(result_b):
        return False
    for ra, rb in zip(result_a, result_b):
        if len(ra) != len(rb):
            return False
        for (ia, sa), (ib, sb) in zip(ra, rb):
            if ia != ib or abs(sa - sb) > tol:
                return False
    return True


def verify_top_k_indices(result_a, result_b):
    """
    Verify two results return the same top-k indices (order-independent).
    """
    if len(result_a) != len(result_b):
        return False
    for ra, rb in zip(result_a, result_b):
        if len(ra) != len(rb):
            return False
        idx_a = set(x[0] for x in ra)
        idx_b = set(x[0] for x in rb)
        if idx_a != idx_b:
            return False
    return True


# ========================================================================
# PLOTTING (Publication-Quality)
# ========================================================================

def plot_bar_comparison(names, values, ylabel, title, save_path, errors=None):
    """
    Generic bar plot with error bars.
    Use for: time comparison, speedup, memory usage.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))

    if errors:
        bars = ax.bar(x, values, yerr=errors, capsize=5,
                     color='steelblue', alpha=0.8, edgecolor='black')
    else:
        bars = ax.bar(x, values, color='steelblue', alpha=0.8, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_speedup(names, times, baseline_name, title="Speedup vs Baseline", save_path="speedup.png"):
    """
    Plot speedup relative to a baseline.
    """
    baseline_time = dict(zip(names, times))[baseline_name]
    speedups = [baseline_time / t for t in times]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, speedups, color='forestgreen', alpha=0.8, edgecolor='black')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_scaling(x_values, y_values_dict, xlabel, ylabel, title, save_path="scaling.png"):
    """
    Line plot with multiple series.
    Use for: scaling with corpus size, workers, or components.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, y_values in y_values_dict.items():
        ax.plot(x_values, y_values, 'o-', label=label, linewidth=2, markersize=6)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_amdahl(worker_counts, speedups, title="Does Amdahl's Law Fit?", save_path="amdahl.png"):
    """
    Plot measured speedup vs. Amdahl's Law prediction.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Measured data
    ax.plot(worker_counts, speedups, 'o-', label='Measured', linewidth=2, markersize=8)

    # Amdahl curves
    w_range = np.linspace(1, max(worker_counts), 100)
    for P in [0.5, 0.75, 0.9, 0.99]:
        amdahl = 1 / ((1 - P) + P / w_range)
        ax.plot(w_range, amdahl, '--', alpha=0.4, label=f"Amdahl P={P}")

    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


# ========================================================================
# TABLES & DATA EXPORT
# ========================================================================

def print_comparison_table(results_dict, baseline_name="baseline"):
    """
    Print a formatted ASCII table.
    results_dict: {version_name: benchmark_result}
    """
    baseline_time = results_dict.get(baseline_name, {}).get("mean", 1.0)

    print("\n" + "=" * 80)
    print(f"{'Version':<25} {'Time (s)':<12} {'Speedup':<10} {'Std (s)':<12} {'Memory (MB)':<12}")
    print("=" * 80)
    for name, res in results_dict.items():
        mean = res.get("mean", res.get("time", 0))
        std = res.get("std", 0)
        mem = res.get("peak_memory_mb", "--")
        speedup = baseline_time / mean if mean > 0 else 0
        mem_str = f"{mem:.1f}" if isinstance(mem, (int, float)) else str(mem)
        print(f"{name:<25} {mean:<12.4f} {speedup:<10.2f}x {std:<12.4f} {mem_str:<12}")
    print("=" * 80)


def save_results_csv(results_dict, filepath="benchmark_results.csv"):
    """Save benchmark results to CSV for easy report inclusion."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['version', 'mean_time', 'std_time', 'min_time', 'max_time', 'peak_memory_mb'])
        for name, res in results_dict.items():
            writer.writerow([
                name,
                res.get('mean', res.get('time', '')),
                res.get('std', ''),
                res.get('min', ''),
                res.get('max', ''),
                res.get('peak_memory_mb', '')
            ])
    print(f"Saved: {filepath}")


# ========================================================================
# PROFILING HELPERS
# ========================================================================

def profile_with_cprofile(func, *args, **kwargs):
    """Run cProfile and print top 20 functions by cumulative time."""
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(20)
    print(s.getvalue())
    return result
