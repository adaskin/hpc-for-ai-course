"""
Phase 3: Ray Strategy (Tasks or Actors)

When to use this: distributing many independent tasks, or stateful workers.

The story: you've exhausted single-machine parallelism. Now you want
to think about scaling beyond one machine. Ray lets you do that
with minimal code changes.

Key question: is the overhead worth it for your dataset size?
"""

import numpy as np
import time
import sys
sys.path.append('..')
from benchmark_utils import benchmark


def search_ray_tasks(queries, corpus, k=5):
    """
    Distribute queries across Ray workers using @ray.remote.
    Uses ray.put() to share corpus in the object store (zero-copy).
    """
    import ray
    ray.init(ignore_reinit_error=True)

    @ray.remote
    def search_one(query, corpus_ref, k):
        corpus = ray.get(corpus_ref)
        similarities = query @ corpus.T
        top_k_idx = np.argpartition(-similarities, k)[:k]
        top_k_scores = similarities[top_k_idx]
        sorted_order = np.argsort(-top_k_scores)
        return list(zip(top_k_idx[sorted_order], top_k_scores[sorted_order]))

    # Share corpus in object store (zero-copy across workers)
    corpus_ref = ray.put(corpus)

    # Submit all tasks
    futures = [search_one.remote(queries[i], corpus_ref, k)
               for i in range(len(queries))]

    # Collect results
    results = ray.get(futures)
    ray.shutdown()
    return results


def search_ray_actors(queries, corpus, k=5, num_actors=4):
    """
    Stateful search actors. Each actor loads corpus once and serves queries.
    Useful for model serving scenarios where setup is expensive.
    """
    import ray
    ray.init(ignore_reinit_error=True)

    @ray.remote
    class SearchActor:
        def __init__(self, corpus):
            self.corpus = corpus
            self.query_count = 0

        def search(self, query, k):
            self.query_count += 1
            similarities = query @ self.corpus.T
            top_k_idx = np.argpartition(-similarities, k)[:k]
            top_k_scores = similarities[top_k_idx]
            sorted_order = np.argsort(-top_k_scores)
            return list(zip(top_k_idx[sorted_order], top_k_scores[sorted_order]))

        def get_stats(self):
            return {"queries_served": self.query_count}

    # Create actors
    actors = [SearchActor.remote(corpus) for _ in range(num_actors)]

    # Distribute queries round-robin
    futures = []
    for i, query in enumerate(queries):
        actor = actors[i % num_actors]
        futures.append(actor.search.remote(query, k))

    results = ray.get(futures)

    # Stats
    stats = ray.get([a.get_stats.remote() for a in actors])
    print(f"Actor stats: {stats}")

    ray.shutdown()
    return results


def search_dask(queries, corpus, k=5):
    """
    Alternative: Dask Array for out-of-core computation.
    Useful when the similarity matrix doesn't fit in RAM.
    """
    import dask.array as da

    queries_da = da.from_array(queries, chunks=(10, queries.shape[1]))
    corpus_da = da.from_array(corpus, chunks=(1000, corpus.shape[1]))

    similarities = queries_da @ corpus_da.T
    similarities_computed = similarities.compute()

    results = []
    for i in range(len(queries)):
        top_k_idx = np.argpartition(-similarities_computed[i], k)[:k]
        top_k_scores = similarities_computed[i, top_k_idx]
        sorted_order = np.argsort(-top_k_scores)
        results.append(list(zip(top_k_idx[sorted_order], top_k_scores[sorted_order])))

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: Ray / Dask Strategy")
    print("=" * 60)

    try:
        corpus = np.load("../corpus_embeddings.npy")
        queries = np.load("../query_embeddings.npy")
    except FileNotFoundError:
        print("Embeddings not found. Generating synthetic data...")
        np.random.seed(42)
        corpus = np.random.randn(5000, 128).astype(np.float32)
        queries = np.random.randn(100, 128).astype(np.float32)
        corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    print(f"Corpus: {corpus.shape}, Queries: {queries.shape}")

    # Benchmark Ray Tasks
    print("\n--- Ray Tasks ---")
    t0 = time.perf_counter()
    _ = search_ray_tasks(queries, corpus, k=5)
    t1 = time.perf_counter()
    print(f"Ray Tasks: {t1 - t0:.3f}s")

    # Benchmark Ray Actors
    print("\n--- Ray Actors ---")
    t0 = time.perf_counter()
    _ = search_ray_actors(queries, corpus, k=5, num_actors=4)
    t1 = time.perf_counter()
    print(f"Ray Actors: {t1 - t0:.3f}s")

    print("\n" + "=" * 60)
    print("YOUR TASKS:")
    print("1. Compare Ray with ProcessPoolExecutor from previous file")
    print("2. Show ray.put() reduces data transfer overhead")
    print("3. When do Actors make sense vs. stateless Tasks?")
    print("4. Is the distributed overhead worth it for your data size?")
    print("5. (Optional) Try Dask for out-of-core arrays")
    print("=" * 60)
