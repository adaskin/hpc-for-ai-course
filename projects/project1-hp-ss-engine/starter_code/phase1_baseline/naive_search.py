"""
Phase 1: Naive Baseline Search

This is your starting point -- intentionally slow pure Python.
Your job: profile it, understand why it's slow, then optimise.

The story starts here: you have a corpus and queries. You need top-k similar docs.
"""

import numpy as np
import time


def cosine_similarity_python(a, b):
    """
    Compute cosine similarity between two vectors using pure Python.
    NO NumPy inside this function -- this is the bottleneck you'll fix later.
    """
    # TODO: implement pure Python cosine similarity
    # Hint: dot product / (norm_a * norm_b)
    raise NotImplementedError("Implement this yourself!")


def search_naive(queries, corpus, k=5):
    """
    For each query, find the top-k most similar documents in corpus.

    Args:
        queries: list of lists (or list of arrays) -- query vectors
        corpus: list of lists (or list of arrays) -- document vectors
        k: number of top results to return

    Returns:
        list of lists: top-k (doc_index, similarity_score) for each query
    """
    results = []
    for q_idx, query in enumerate(queries):
        scores = []
        for d_idx, doc in enumerate(corpus):
            sim = cosine_similarity_python(query, doc)
            scores.append((d_idx, sim))

        # Sort by score descending, take top k
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append(scores[:k])

    return results


if __name__ == "__main__":
    # Load your dataset here
    # Replace with your actual data loading
    print("Loading embeddings...")
    try:
        corpus = np.load("corpus_embeddings.npy").tolist()
        queries = np.load("query_embeddings.npy").tolist()
    except FileNotFoundError:
        print("Embeddings not found. Generate your dataset first!")
        print("See the project guidelines for dataset options.")
        exit(1)

    print(f"Corpus: {len(corpus)} documents x {len(corpus[0])} dimensions")
    print(f"Queries: {len(queries)} queries")

    # Baseline benchmark
    print("\nRunning naive baseline...")
    t0 = time.perf_counter()
    results = search_naive(queries, corpus, k=5)
    t1 = time.perf_counter()

    baseline_time = t1 - t0
    print(f"\nBaseline time: {baseline_time:.3f} seconds")
    print(f"Time per query: {baseline_time / len(queries) * 1000:.1f} ms")
    print(f"First query top-5: {results[0]}")

    # Save baseline time for comparison in later phases
    with open("baseline_time.txt", "w") as f:
        f.write(f"{baseline_time}\n")
    print("Saved baseline_time.txt for comparison in later phases.")

    # TODO: Now profile this with cProfile and line_profiler
    # See profiling_template.py
