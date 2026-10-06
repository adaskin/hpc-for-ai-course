"""
Phase 3: ThreadPoolExecutor Strategy

When to use this: I/O-bound tasks or NumPy/BLAS operations (GIL released).

The story: you've vectorised your search. Now you have multiple queries.
Can you process them in parallel with threads?

Key question: does the GIL block you, or does BLAS release it?
"""

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append('..')
from benchmark_utils import benchmark


def search_single_numpy(query, corpus, k=5):
    """Search one query using NumPy vectorisation."""
    similarities = query @ corpus.T
    top_k_idx = np.argpartition(-similarities, k)[:k]
    top_k_scores = similarities[top_k_idx]
    sorted_order = np.argsort(-top_k_scores)
    return list(zip(top_k_idx[sorted_order], top_k_scores[sorted_order]))


def search_threadpool(queries, corpus, k=5, max_workers=4):
    """
    Process queries in parallel using ThreadPoolExecutor.
    Each query gets its own thread.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(search_single_numpy, queries[i], corpus, k)
            for i in range(len(queries))
        ]
        results = [f.result() for f in futures]
    return results


def benchmark_thread_scaling(queries, corpus, k=5):
    """
    Benchmark with 1, 2, 4, 8 workers.
    Plot speedup vs. workers to see if scaling is linear.
    """
    worker_counts = [1, 2, 4, 8]
    results = {}

    for workers in worker_counts:
        print(f"\nWorkers: {workers}")
        res = benchmark(search_threadpool, queries, corpus, k, workers,
                       warmup=2, runs=5)
        results[workers] = res
        print(f"  Mean: {res['mean']:.4f}s, Std: {res['std']:.4f}s")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: ThreadPoolExecutor Strategy")
    print("=" * 60)

    # Load your embeddings
    try:
        corpus = np.load("../corpus_embeddings.npy")
        queries = np.load("../query_embeddings.npy")
    except FileNotFoundError:
        print("Embeddings not found. Generate your dataset first!")
        exit(1)

    print(f"Corpus: {corpus.shape}, Queries: {queries.shape}")
    print(f"CPU cores: {__import__('os').cpu_count()}")

    # Benchmark
    results = benchmark_thread_scaling(queries, corpus, k=5)

    # Calculate speedup
    baseline_time = results[1]['mean']
    print("\nSpeedup vs single-threaded:")
    for workers, res in results.items():
        speedup = baseline_time / res['mean']
        print(f"  {workers} workers: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("YOUR TASKS:")
    print("1. Does speedup scale linearly with workers?")
    print("2. Why or why not? (GIL, BLAS thread competition)")
    print("3. Compare with pure Python search in threads (should be worse)")
    print("4. Is this strategy worth it for your workload?")
    print("=" * 60)
