"""
Week 4 Part I – GIL demo: I/O-bound (LLM API) vs CPU-bound (tokenization).
Run: python 01_gil_llm_vs_tokenize.py

Shows that threads help for I/O but NOT for CPU-bound Python.
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor


# ── Task A: LLM inference query (I/O-bound) ───────────────────────

def query_llm(prompt_id: int) -> str:
    """Simulate sending a prompt to a vLLM / TGI inference server."""
    time.sleep(0.2)  # simulated network + generation latency
    return f"response_{prompt_id}"


def demo_io_bound():
    n_prompts = 20
    n_workers = 10

    # Serial
    t0 = time.perf_counter()
    serial = [query_llm(i) for i in range(n_prompts)]
    t_serial = time.perf_counter() - t0

    # Threads
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        parallel = list(ex.map(query_llm, range(n_prompts)))
    t_threads = time.perf_counter() - t0

    print("=" * 60)
    print("TASK A: LLM Inference (I/O-bound, simulated)")
    print(f"  Prompts: {n_prompts}, Workers: {n_workers}")
    print(f"  Serial:   {t_serial:.2f} s")
    print(f"  Threads:  {t_threads:.2f} s")
    print(f"  Speedup:  {t_serial / t_threads:.1f}×  ← threads HELP")
    print()


# ── Task B: Pure-Python tokenizer (CPU-bound) ─────────────────────

def tokenize_pure_python(text: str) -> int:
    """Char-level tokenizer in pure Python. CPU-bound. GIL-bound."""
    tokens = []
    for char in text:
        if char == ' ':
            tokens.append('<SEP>')
        else:
            tokens.append(char)
    return len(tokens)


def demo_cpu_bound():
    text = "the transformer model uses self attention to process sequences " * 2000
    documents = [text] * 8
    n_workers = 4

    # Serial
    t0 = time.perf_counter()
    serial = [tokenize_pure_python(d) for d in documents]
    t_serial = time.perf_counter() - t0

    # Threads (GIL blocks!)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        threaded = list(ex.map(tokenize_pure_python, documents))
    t_threads = time.perf_counter() - t0

    print("=" * 60)
    print("TASK B: Pure-Python Tokenizer (CPU-bound)")
    print(f"  Documents: {len(documents)}, Workers: {n_workers}")
    print(f"  Serial:   {t_serial:.2f} s")
    print(f"  Threads:  {t_threads:.2f} s")
    print(f"  Speedup:  {t_serial / t_threads:.2f}×  ← threads DON'T help (GIL!)")
    print()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    print(f"Python: {sys.version.split()[0]}")
    print(f"GIL switch interval: {sys.getswitchinterval()} s")
    print(f"CPU cores: {os.cpu_count()}")
    print()

    demo_io_bound()
    demo_cpu_bound()

    print("=" * 60)
    print("CONCLUSION:")
    print("  I/O-bound → threads help (GIL released during wait)")
    print("  CPU-bound Python → threads DON'T help (GIL held)")
    print("  Fix for CPU-bound: use ProcessPoolExecutor (next demo)")