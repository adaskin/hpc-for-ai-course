"""
Phase 2B: Dimensionality Reduction -- Choose and Justify

Your corpus might have 5K-20K documents with 768-dim embeddings.
That's a lot of numbers to search through. Dimensionality reduction
is your first real speedup -- but which method should you choose?

This is a design decision. Your report must justify your choice with data.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA


def power_iteration_svd(X, k, max_iter=100, tol=1e-6):
    """
    Compute top-k singular vectors using power iteration.

    This is YOUR implementation. The core algorithm:
    1. Start with random orthonormal V (n x k)
    2. Iterate: V_new = qr(X^T @ X @ V)
    3. Stop when converged
    4. Compute U and S from XV

    Args:
        X: (m x n) matrix
        k: number of components to keep
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        U_k (m x k), S_k (k,), V_k^T (k x n)
    """
    # TODO: implement power iteration SVD
    # Hints:
    # - Use np.linalg.qr for orthonormalisation
    # - Check convergence with np.linalg.norm(V - V_new)
    # - Compute final U, S from X @ V
    raise NotImplementedError("Implement power iteration SVD yourself!")


def reconstruction_error(X, U, S, Vt):
    """
    Compute ||X - U @ diag(S) @ Vt||_F (Frobenius norm).
    Lower is better.
    """
    X_recon = U @ np.diag(S) @ Vt
    return np.linalg.norm(X - X_recon)


def compare_svd_methods(X, k_values):
    """
    Compare your power iteration against library implementations.

    Methods to compare (choose at least 2 library methods):
    - Your power_iteration_svd
    - np.linalg.svd (full SVD)
    - sklearn TruncatedSVD
    - sklearn PCA
    - sklearn IncrementalPCA

    Returns:
        dict mapping method_name -> list of (k, time, error) tuples
    """
    results = {}

    # TODO: benchmark each method for each k in k_values
    # For each:
    #   1. Time the execution
    #   2. Compute reconstruction error
    #   3. Record peak memory if possible

    print("TODO: implement comparison across methods")
    return results


def plot_svd_comparison(results):
    """
    Create 2 subplots:
    - Time vs. k
    - Reconstruction error vs. k

    Each line is a different method.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TODO: plot each method's results
    # Use different colors/markers for each method
    # Label axes, add legend, save figure

    plt.tight_layout()
    plt.savefig('svd_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: svd_comparison.png")


def plot_scaling_comparison(matrix_sizes, k=50):
    """
    Plot time vs. matrix size for your SVD vs. library methods.
    Shows how each method scales.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # TODO: benchmark at each matrix size, plot results

    ax.set_xlabel('Matrix Rows (m)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Scaling with Matrix Size (k={k})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('svd_scaling.png', dpi=150, bbox_inches='tight')
    print("Saved: svd_scaling.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2B: Dimensionality Reduction")
    print("=" * 60)

    # Generate test data (replace with your actual embeddings)
    np.random.seed(42)
    m, n = 5000, 768
    X = np.random.randn(m, n).astype(np.float32)
    print(f"Test matrix: {m} x {n}")

    # First: verify your power iteration is correct
    print("\nVerifying your power iteration SVD...")
    k_test = 50
    U_pi, S_pi, Vt_pi = power_iteration_svd(X, k_test)
    print(f"Shapes: U={U_pi.shape}, S={S_pi.shape}, Vt={Vt_pi.shape}")

    # Compare against numpy (ground truth)
    U_np, S_np, Vt_np = np.linalg.svd(X, full_matrices=False)
    err_yours = reconstruction_error(X, U_pi, S_pi, Vt_pi)
    err_numpy = reconstruction_error(X, U_np[:, :k_test], S_np[:k_test], Vt_np[:k_test, :])
    print(f"Reconstruction error -- yours: {err_yours:.2f}, numpy: {err_numpy:.2f}")

    # Benchmark across k values
    k_values = [10, 25, 50, 100, 200]
    print(f"\nBenchmarking for k = {k_values}...")
    results = compare_svd_methods(X, k_values)
    plot_svd_comparison(results)

    # Benchmark scaling
    print("\nBenchmarking scaling with matrix size...")
    matrix_sizes = [(1000, 768), (2500, 768), (5000, 768), (10000, 768)]
    plot_scaling_comparison(matrix_sizes, k=50)

    print("\n" + "=" * 60)
    print("YOUR TASKS:")
    print("1. Implement power_iteration_svd correctly")
    print("2. Compare against at least 2 library methods")
    print("3. Plot time vs. k and error vs. k")
    print("4. Make a design decision: which method for your final system?")
    print("5. Justify with data: time, error, and memory trade-offs")
    print("=" * 60)
