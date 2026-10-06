"""
Phase 3: ProcessPoolExecutor Strategy

When to use this: CPU-bound pure Python (GIL blocks threads).

The story: threads didn't help because of the GIL.
Processes bypass the GIL entirely. But there's a cost: pickling.

Key question: how do you share data between processes without
paying the pickling penalty?
"""

import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.append('..')
from benchmark_utils import benchmark


def search_batch_python(query_batch, corpus_list, k=5):
    """
    Search a batch of queries using pure Python.
    This function runs in a separate process.
    """
    results = []
    for query in query_batch:
        scores = []
        for d_idx, doc in enumerate(corpus_list):
            # Pure Python cosine similarity
            dot = 0.0
            norm_q = 0.0
            norm_d = 0.0
            for i in range(len(query)):
                dot += query[i] * doc[i]
                norm_q += query[i] * query[i]
                norm_d += doc[i] * doc[i]
            sim = dot / (norm_q ** 0.5 * norm_d ** 0.5) if norm_q > 0 and norm_d > 0 else 0.0
            scores.append((d_idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append(scores[:k])
    return results


def search_processpool_bad(queries_list, corpus_list, k=5, max_workers=4):
    """
    BAD approach: submit one task per query, pass corpus every time.
    This pickles the entire corpus for each query. Very slow.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(search_batch_python, [queries_list[i]], corpus_list, k)
            for i in range(len(queries_list))
        ]
        results = [f.result()[0] for f in futures]
    return results


def search_processpool_good(queries_list, corpus_list, k=5, max_workers=4):
    """
    GOOD approach: pass indices, not data. Or batch queries per process.
    Each worker gets a chunk of queries, reducing pickling overhead.
    """
    # Split queries into batches
    batch_size = (len(queries_list) + max_workers - 1) // max_workers
    batches = [
        queries_list[i:i + batch_size]
        for i in range(0, len(queries_list), batch_size)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(search_batch_python, batch, corpus_list, k)
            for batch in batches
        ]
        results = []
        for f in futures:
            results.extend(f.result())
    return results


def search_processpool_numpy(queries, corpus, k=5, max_workers=4):
    """
    ProcessPool with NumPy (pass indices, use shared memory if possible).
    For NumPy arrays, the "good" approach is to pass indices and
    let each worker slice the shared array.
    """
    # TODO: implement the index-passing approach
    # Each worker gets a range of query indices
    raise NotImplementedError("Implement index-passing version")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: ProcessPoolExecutor Strategy")
    print("=" * 60)

    try:
        corpus = np.load("../corpus_embeddings.npy")
        queries = np.load("../query_embeddings.npy")
    except FileNotFoundError:
        print("Embeddings not found. Generate your dataset first!")
        exit(1)

    queries_list = queries.tolist()
    corpus_list = corpus.tolist()

    print(f"Corpus: {corpus.shape}, Queries: {queries.shape}")

    # Compare bad vs good approach
    print("\n--- Bad approach (pickle per query) ---")
    t0 = time.perf_counter()
    _ = search_processpool_bad(queries_list[:20], corpus_list, k=5, max_workers=4)
    t1 = time.perf_counter()
    print(f"Bad approach: {t1 - t0:.3f}s (only 20 queries!)")

    print("\n--- Good approach (batched) ---")
    t0 = time.perf_counter()
    _ = search_processpool_good(queries_list, corpus_list, k=5, max_workers=4)
    t1 = time.perf_counter()
    print(f"Good approach: {t1 - t0:.3f}s (all queries)")

    print("\n" + "=" * 60)
    print("YOUR TASKS:")
    print("1. Show the pickling overhead problem with data")
    print("2. Implement the batched/index-passing fix")
    print("3. Benchmark speedup vs. number of workers")
    print("4. Compare with ThreadPoolExecutor from previous file")
    print("5. Discuss: when is ProcessPool worth the overhead?")
    print("=" * 60)
