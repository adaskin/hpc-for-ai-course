"""
Phase 2A: Sparse vs Dense -- The Representation Decision

Before you search fast, you need to decide: how do you store your data?
Sparse (TF-IDF, CSR) or dense (embeddings)?

This is a design decision with real performance consequences.
Your report should justify which representation you chose for your final system.
"""

import numpy as np
import time
import tracemalloc
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def memory_usage(array_or_matrix):
    """Return memory usage in MB."""
    if hasattr(array_or_matrix, 'data'):
        # Sparse matrix
        return (array_or_matrix.data.nbytes +
                array_or_matrix.indices.nbytes +
                array_or_matrix.indptr.nbytes) / (1024 * 1024)
    else:
        # Dense array
        return array_or_matrix.nbytes / (1024 * 1024)


def sparse_dot_sparse(A_sparse, B_sparse):
    """
    Compute A @ B.T where both are sparse CSR matrices.
    TODO: implement using scipy.sparse operations, then try manually.
    """
    # TODO: your implementation here
    raise NotImplementedError("Implement sparse-sparse dot product")


def sparse_dot_dense(A_sparse, B_dense):
    """
    Compute A @ B.T where A is sparse CSR and B is dense.
    TODO: implement manually (iterate over nonzero elements of A).
    """
    # TODO: your implementation here
    raise NotImplementedError("Implement sparse-dense dot product")


def dense_dot_dense(A_dense, B_dense):
    """
    Compute A @ B.T where both are dense.
    TODO: implement with NumPy first, then try pure Python loops for comparison.
    """
    # TODO: your implementation here
    raise NotImplementedError("Implement dense-dense dot product")


def compare_representations(corpus_texts, query_texts, vectorizer, svd):
    """
    Compare sparse and dense representations at different corpus sizes.

    Returns:
        list of dicts with corpus_size, sparse_mem, dense_mem,
        sparse_sparse_time, sparse_dense_time, dense_dense_time
    """
    corpus_sizes = [1000, 2500, 5000, 10000]
    results = []

    for size in corpus_sizes:
        if size > len(corpus_texts):
            break

        subset = corpus_texts[:size]
        print(f"\n--- Corpus size: {size} ---")

        # Build sparse matrix
        X_sparse = vectorizer.transform(subset)
        print(f"Sparse shape: {X_sparse.shape}, nnz: {X_sparse.nnz}, "
              f"density: {X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.4f}")

        # Build dense matrix
        X_dense = svd.transform(X_sparse)
        print(f"Dense shape: {X_dense.shape}")

        # Memory comparison
        sparse_mem = memory_usage(X_sparse)
        dense_mem = memory_usage(X_dense)
        print(f"Memory: sparse={sparse_mem:.2f}MB, dense={dense_mem:.2f}MB, "
              f"ratio={dense_mem/sparse_mem:.1f}x")

        # Query representations
        Q_sparse = vectorizer.transform(query_texts)
        Q_dense = svd.transform(Q_sparse)

        # TODO: benchmark dot products for each combination
        # Record times, store in results

        results.append({
            "corpus_size": size,
            "sparse_mem_mb": sparse_mem,
            "dense_mem_mb": dense_mem,
            # TODO: fill in timing results
        })

    return results


def plot_sparse_dense(results):
    """
    Plot memory and compute comparison.
    TODO: create 2 subplots: memory vs corpus size, time vs corpus size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sizes = [r["corpus_size"] for r in results]

    # Memory plot
    sparse_mems = [r["sparse_mem_mb"] for r in results]
    dense_mems = [r["dense_mem_mb"] for r in results]
    axes[0].plot(sizes, sparse_mems, 'o-', label='Sparse (CSR)', linewidth=2)
    axes[0].plot(sizes, dense_mems, 's-', label='Dense', linewidth=2)
    axes[0].set_xlabel('Corpus Size (documents)')
    axes[0].set_ylabel('Memory (MB)')
    axes[0].set_title('Memory Footprint')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # TODO: add compute time comparison subplot

    plt.tight_layout()
    plt.savefig('sparse_dense_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: sparse_dense_comparison.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2A: Sparse vs Dense Comparison")
    print("=" * 60)

    # TODO: replace with your actual dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    # Placeholder: generate synthetic data
    np.random.seed(42)
    vocab = [f"word_{i}" for i in range(5000)]
    corpus_texts = [
        " ".join(np.random.choice(vocab, size=np.random.randint(50, 200)))
        for _ in range(10000)
    ]
    query_texts = [
        " ".join(np.random.choice(vocab, size=np.random.randint(20, 50)))
        for _ in range(100)
    ]

    # Fit on full corpus
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_full = vectorizer.fit_transform(corpus_texts)

    svd = TruncatedSVD(n_components=128, random_state=42)
    svd.fit(X_full)

    results = compare_representations(corpus_texts, query_texts, vectorizer, svd)
    plot_sparse_dense(results)

    print("\n" + "=" * 60)
    print("YOUR TASKS:")
    print("1. Implement all three dot products (sparse-sparse, sparse-dense, dense-dense)")
    print("2. Benchmark and plot memory + compute time")
    print("3. Make a design decision: sparse or dense for your final system?")
    print("4. Justify with data from your plots")
    print("=" * 60)
