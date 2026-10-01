
"""
Week 4 Part II – Batch LLM inference with ThreadPoolExecutor.
Run: python 02_batch_llm_inference.py

Simulates sending prompts to an inference server (vLLM / TGI).
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def query_inference_server(prompt: str) -> dict:
    """Simulate HTTP POST to http://localhost:8000/generate."""
    time.sleep(0.2)  # simulated latency (network + generation)
    return {"prompt": prompt, "response": f"Generated response for: '{prompt[:40]}'"}


def main():
    prompts = [
        "Explain gradient descent in one sentence.",
        "What is the attention mechanism in transformers?",
        "Write a haiku about backpropagation.",
        "Compare Adam and SGD optimizers briefly.",
        "What is batch normalization and why does it help?",
        "Explain the vanishing gradient problem.",
        "What is transfer learning in deep learning?",
        "Describe the difference between L1 and L2 regularization.",
        "What is dropout and how does it prevent overfitting?",
        "Explain what a residual connection does.",
        "What is the purpose of the softmax function?",
        "Describe the difference between training and inference.",
        "What is a learning rate schedule?",
        "Explain what embedding layers do.",
        "What is the purpose of layer normalization?",
        "Describe the difference between RNNs and transformers.",
        "What is beam search in text generation?",
        "Explain what a loss landscape is.",
        "What is the purpose of data augmentation?",
        "Describe the difference between supervised and self-supervised learning.",
    ]

    print(f"Sending {len(prompts)} prompts to inference server (simulated)")
    print(f"Each request takes ~0.2s\n")

    # ── Serial ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    serial_results = [query_inference_server(p) for p in prompts]
    t_serial = time.perf_counter() - t0
    print(f"Serial:   {t_serial:.2f} s")

    # ── Parallel with executor.map ────────────────────────────────
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as ex:
        map_results = list(ex.map(query_inference_server, prompts))
    t_map = time.perf_counter() - t0
    print(f"map():    {t_map:.2f} s  ({t_serial/t_map:.1f}× speedup)")

    # ── Parallel with submit + as_completed ───────────────────────
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(query_inference_server, p): i for i, p in enumerate(prompts)}
        submit_results = [None] * len(prompts)
        for future in as_completed(futures):
            idx = futures[future]
            submit_results[idx] = future.result()
    t_submit = time.perf_counter() - t0
    print(f"submit(): {t_submit:.2f} s  ({t_serial/t_submit:.1f}× speedup)")

    print(f"\nGot {len(map_results)} responses. First:")
    print(f"  Prompt:   {map_results[0]['prompt']}")
    print(f"  Response: {map_results[0]['response']}")


if __name__ == "__main__":
    main()