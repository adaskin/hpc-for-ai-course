"""
Phase 3: Async I/O Strategy

When to use this: massive concurrent I/O (>100 tasks, e.g., batch API queries).

The story: your search is now fast. But what if queries come in
as async HTTP requests? Or what if you need to call an embedding API?

Key question: does async help CPU-bound search, or only I/O-bound retrieval?
"""

import numpy as np
import time
import asyncio
import sys
sys.path.append('..')
from benchmark_utils import benchmark


async def get_embedding_async(text, delay=0.01):
    """
    Simulate an async embedding API call.
    In a real system, this would be an HTTP request to an embedding service.
    """
    await asyncio.sleep(delay)  # network latency
    return np.random.randn(128).astype(np.float32)


async def search_api_async(query_text, corpus, k=5, embed_delay=0.01):
    """
    Full async pipeline:
    1. Async embedding retrieval (I/O)
    2. CPU-bound similarity (blocks event loop!)
    3. Return top-k
    """
    # Step 1: async I/O
    query_embedding = await get_embedding_async(query_text, delay=embed_delay)

    # Step 2: CPU work (this blocks!)
    similarities = query_embedding @ corpus.T
    top_k_idx = np.argpartition(-similarities, k)[:k]
    top_k_scores = similarities[top_k_idx]
    sorted_order = np.argsort(-top_k_scores)

    return list(zip(top_k_idx[sorted_order], top_k_scores[sorted_order]))


async def search_all_serial(query_texts, corpus, k=5, embed_delay=0.01):
    """Process queries one at a time (serial async)."""
    results = []
    for text in query_texts:
        result = await search_api_async(text, corpus, k, embed_delay)
        results.append(result)
    return results


async def search_all_concurrent(query_texts, corpus, k=5, embed_delay=0.01):
    """Process all queries concurrently with asyncio.gather."""
    tasks = [
        search_api_async(text, corpus, k, embed_delay)
        for text in query_texts
    ]
    return await asyncio.gather(*tasks)


async def search_all_bounded(query_texts, corpus, k=5, embed_delay=0.01, max_concurrent=10):
    """
    Bounded concurrency with semaphore.
    Prevents overwhelming the API.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_search(text):
        async with semaphore:
            return await search_api_async(text, corpus, k, embed_delay)

    tasks = [bounded_search(text) for text in query_texts]
    return await asyncio.gather(*tasks)


def benchmark_async(query_texts, corpus, k=5, embed_delay=0.01):
    """Benchmark serial vs concurrent vs bounded async."""
    results = {}

    # Serial
    t0 = time.perf_counter()
    _ = asyncio.run(search_all_serial(query_texts, corpus, k, embed_delay))
    t1 = time.perf_counter()
    results["serial"] = t1 - t0
    print(f"Serial:      {t1 - t0:.3f}s")

    # Concurrent
    t0 = time.perf_counter()
    _ = asyncio.run(search_all_concurrent(query_texts, corpus, k, embed_delay))
    t1 = time.perf_counter()
    results["concurrent"] = t1 - t0
    print(f"Concurrent:  {t1 - t0:.3f}s")

    # Bounded
    t0 = time.perf_counter()
    _ = asyncio.run(search_all_bounded(query_texts, corpus, k, embed_delay, max_concurrent=10))
    t1 = time.perf_counter()
    results["bounded"] = t1 - t0
    print(f"Bounded:     {t1 - t0:.3f}s")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: Async I/O Strategy")
    print("=" * 60)

    try:
        corpus = np.load("../corpus_embeddings.npy")
    except FileNotFoundError:
        print("Embeddings not found. Generating synthetic data...")
        np.random.seed(42)
        corpus = np.random.randn(5000, 128).astype(np.float32)
        corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

    query_texts = [f"Query about ML topic {i}" for i in range(100)]

    print(f"Corpus: {corpus.shape}")
    print(f"Queries: {len(query_texts)}")

    for delay in [0.01, 0.05]:
        print(f"\n--- Simulated API latency: {delay}s ---")
        results = benchmark_async(query_texts, corpus, k=5, embed_delay=delay)
        speedup = results["serial"] / results["concurrent"]
        print(f"Speedup (concurrent vs serial): {speedup:.1f}x")

    print("\n" + "=" * 60)
    print("YOUR TASKS:")
    print("1. Does async help when the API latency is the bottleneck?")
    print("2. What if the embedding is pre-computed (no I/O)?")
    print("3. When would bounded concurrency matter?")
    print("4. How would you combine async I/O with ThreadPool for CPU work?")
    print("=" * 60)
