"""
Week 4 Part II – Tokenization: Threads vs Processes.
Run: python 04_parallel_tokenization.py

Shows that threads DON'T help for CPU-bound Python (GIL),
but processes DO.
"""
import time
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def tokenize_python(text: str) -> int:
    """Pure-Python word-level tokenizer. CPU-bound. GIL-bound."""
    vocab = {}
    tokens = []
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)
        tokens.append(vocab[word])
    return len(tokens)


def main():
    # Create documents
    base = "transformer attention mechanism gradient descent optimizer "
    base += "embedding layer normalization softmax cross entropy loss "
    documents = [base * 3000] * 8  # 8 documents, each ~150k chars

    n_workers = min(4, os.cpu_count() or 4)

    print(f"Documents: {len(documents)}, Workers: {n_workers}")
    print(f"Each doc: ~{len(documents[0]):,} chars\n")

    # ── Serial ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    serial = [tokenize_python(d) for d in documents]
    t_serial = time.perf_counter() - t0
    print(f"Serial:     {t_serial:.2f} s")

    # ── Threads (GIL blocks) ──────────────────────────────────────
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        threaded = list(ex.map(tokenize_python, documents))
    t_threads = time.perf_counter() - t0
    print(f"Threads:    {t_threads:.2f} s  (speedup: {t_serial/t_threads:.2f}× ← GIL!)")

    # ── Processes (own GIL) ───────────────────────────────────────
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        processed = list(ex.map(tokenize_python, documents))
    t_procs = time.perf_counter() - t0
    print(f"Processes:  {t_procs:.2f} s  (speedup: {t_serial/t_procs:.2f}× ✓)")

    print(f"\nConclusion: Processes give {t_serial/t_procs:.1f}× speedup.")
    print(f"            Threads give {t_serial/t_threads:.2f}× (GIL blocks parallelism).")


if __name__ == "__main__":
    main()