"""
Week 4 Part V – Combined lab: all four AI workload experiments.
Run: python 09_lab_all.py

Experiments:
  1. LLM batch inference (I/O-bound): Serial vs Threads vs Async
  2. Corpus tokenization (CPU-bound): Threads vs Processes
  3. Batch augmentation (NumPy): Serial vs Threads
  4. Agent swarm: Async
"""
import asyncio
import time
import os
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: LLM Batch Inference (I/O-bound)
# ══════════════════════════════════════════════════════════════════

def query_llm_sync(prompt: str) -> str:
    time.sleep(0.2)
    return f"resp_{prompt[:10]}"

async def query_llm_async(prompt: str) -> str:
    await asyncio.sleep(0.2)
    return f"resp_{prompt[:10]}"

def experiment_1():
    prompts = [f"prompt_{i}" for i in range(30)]
    n_workers = 10

    print("=" * 60)
    print("EXPERIMENT 1: LLM Batch Inference (30 prompts, 200ms each)")

    # Serial
    t0 = time.perf_counter()
    [query_llm_sync(p) for p in prompts]
    t_serial = time.perf_counter() - t0

    # Threads
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(query_llm_sync, prompts))
    t_threads = time.perf_counter() - t0

    # Async
    async def run_async():
        tasks = [query_llm_async(p) for p in prompts]
        await asyncio.gather(*tasks)

    t0 = time.perf_counter()
    asyncio.run(run_async())
    t_async = time.perf_counter() - t0

    print(f"  Serial:   {t_serial:.2f} s")
    print(f"  Threads:  {t_threads:.2f} s  ({t_serial/t_threads:.1f}×)")
    print(f"  Async:    {t_async:.2f} s  ({t_serial/t_async:.1f}×)")
    print()


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Corpus Tokenization (CPU-bound, pure Python)
# ══════════════════════════════════════════════════════════════════

def tokenize_python(text: str) -> int:
    vocab = {}
    tokens = []
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)
        tokens.append(vocab[word])
    return len(tokens)

def experiment_2():
    base = "transformer attention gradient descent optimizer embedding " * 3000
    documents = [base] * 8
    n_workers = min(4, os.cpu_count() or 4)

    print("=" * 60)
    print(f"EXPERIMENT 2: Tokenization (8 docs, {n_workers} workers)")

    # Serial
    t0 = time.perf_counter()
    [tokenize_python(d) for d in documents]
    t_serial = time.perf_counter() - t0

    # Threads (GIL!)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(tokenize_python, documents))
    t_threads = time.perf_counter() - t0

    # Processes
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(tokenize_python, documents))
    t_procs = time.perf_counter() - t0

    print(f"  Serial:    {t_serial:.2f} s")
    print(f"  Threads:   {t_threads:.2f} s  ({t_serial/t_threads:.2f}× ← GIL!)")
    print(f"  Processes: {t_procs:.2f} s  ({t_serial/t_procs:.2f}× ✓)")
    print()


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Batch Augmentation (NumPy, GIL released)
# ══════════════════════════════════════════════════════════════════

def augment_image(img: np.ndarray) -> np.ndarray:
    if np.random.rand() > 0.5:
        img = img[:, ::-1, :].copy()
    img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
    return img

def experiment_3():
    batch = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(32)]
    n_workers = 8

    print("=" * 60)
    print(f"EXPERIMENT 3: Batch Augmentation (32 images, {n_workers} threads)")

    # Serial
    t0 = time.perf_counter()
    [augment_image(img) for img in batch]
    t_serial = time.perf_counter() - t0

    # Threads (NumPy releases GIL)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(augment_image, batch))
    t_threads = time.perf_counter() - t0

    print(f"  Serial:   {t_serial:.4f} s")
    print(f"  Threads:  {t_threads:.4f} s  ({t_serial/t_threads:.1f}× ← NumPy releases GIL)")
    print()


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Agent Swarm (Async)
# ══════════════════════════════════════════════════════════════════

async def call_tool(agent_id: int, tool: str) -> str:
    await asyncio.sleep(random.uniform(0.1, 0.3))
    return f"A{agent_id}:{tool}"

async def agent_loop(agent_id: int) -> list:
    tools = ["search", "calc", "code"]
    return await asyncio.gather(*[call_tool(agent_id, t) for t in tools])

def experiment_4():
    n_agents = 10
    print("=" * 60)
    print(f"EXPERIMENT 4: Agent Swarm ({n_agents} agents × 3 tools = {n_agents*3} calls)")

    async def run():
        agents = [agent_loop(i) for i in range(n_agents)]
        results = await asyncio.gather(*agents)
        return results

    t0 = time.perf_counter()
    results = asyncio.run(run())
    t_async = time.perf_counter() - t0

    total_calls = sum(len(r) for r in results)
    print(f"  Async: {t_async:.2f} s  ({total_calls} calls, 1 thread)")
    print(f"  Serial would be: ~{total_calls * 0.2:.1f} s")
    print()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Python: {os.sys.version.split()[0]}\n")

    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()

    print("=" * 60)
    print("DONE. Fill in the results table and answer discussion questions.")