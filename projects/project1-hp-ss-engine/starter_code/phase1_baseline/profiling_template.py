"""
Phase 1: Profiling Template

Use this to profile your naive search and identify bottlenecks.
The story: before you fix anything, you need to know what's broken.
"""

import cProfile
import pstats
import io
from line_profiler import LineProfiler

# Import your naive search
from naive_search import search_naive, cosine_similarity_python
import numpy as np


def profile_with_cprofile(queries, corpus, k=5):
    """
    Profile the search function with cProfile.
    Shows which functions take the most cumulative time.
    """
    print("=" * 60)
    print("cProfile Output (sorted by cumulative time)")
    print("=" * 60)

    pr = cProfile.Profile()
    pr.enable()
    results = search_naive(queries, corpus, k)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(20)  # top 20 functions
    print(s.getvalue())

    return results


def profile_with_line_profiler(queries, corpus, k=5):
    """
    Profile line-by-line with line_profiler.
    Shows which lines inside cosine_similarity_python are slowest.

    Requires: pip install line_profiler
    Usage: kernprof -l -v profiling_template.py
    """
    print("=" * 60)
    print("Line Profiler Output")
    print("=" * 60)

    # Decorate the target function
    profiled_cosine = profile(cosine_similarity_python)

    # Run a small subset (line_profiler is slow)
    small_queries = queries[:10]
    small_corpus = corpus[:1000]

    for query in small_queries:
        for doc in small_corpus:
            profiled_cosine(query, doc)


def profile_with_pyspy():
    """
    Generate a flame graph with py-spy.

    Run from terminal:
        py-spy record -o flamegraph.svg -- python naive_search.py

    This gives you a visual flame graph showing where time is spent.
    """
    print("Run this from your terminal:")
    print("  py-spy record -o flamegraph.svg -- python naive_search.py")
    print("Then open flamegraph.svg in your browser.")


if __name__ == "__main__":
    # Load data
    try:
        corpus = np.load("corpus_embeddings.npy").tolist()
        queries = np.load("query_embeddings.npy").tolist()
    except FileNotFoundError:
        print("Embeddings not found. Generate your dataset first!")
        exit(1)

    print(f"Corpus: {len(corpus)} documents")
    print(f"Queries: {len(queries)} queries")

    # 1. cProfile
    print("\n" + "=" * 60)
    profile_with_cprofile(queries, corpus, k=5)

    # 2. Line profiler (run with: kernprof -l -v profiling_template.py)
    # Uncomment the line below if running via kernprof
    # profile_with_line_profiler(queries, corpus, k=5)

    # 3. py-spy instructions
    print("\n" + "=" * 60)
    profile_with_pyspy()

    print("\n" + "=" * 60)
    print("YOUR TASKS:")
    print("1. Identify the top 3 bottlenecks from cProfile")
    print("2. Run line_profiler to see which lines are slowest")
    print("3. Form a hypothesis: what will give the biggest speedup?")
    print("4. Record your baseline time for later comparison")
    print("=" * 60)
