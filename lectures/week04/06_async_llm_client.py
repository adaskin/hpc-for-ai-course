"""
Week 4 Part III – Async LLM client: batch inference.
Run: python 06_async_llm_client.py

Simulates sending 50 prompts concurrently to an inference server.
One thread. ~KB memory. ~0.2s total.
"""
import asyncio
import time


async def query_inference_server(prompt: str, latency: float = 0.2) -> str:
    """Simulate async HTTP POST to vLLM / TGI at localhost:8000/generate."""
    await asyncio.sleep(latency)  # non-blocking network wait
    return f"Response to: '{prompt[:35]}'"


async def batch_inference(prompts: list[str]) -> list[str]:
    tasks = [query_inference_server(p) for p in prompts]
    return await asyncio.gather(*tasks)


async def main():
    prompts = [f"Explain concept {i} in one sentence." for i in range(50)]

    print(f"Sending {len(prompts)} prompts (simulated, 200ms each)\n")

    # Serial
    t0 = time.perf_counter()
    serial = []
    for p in prompts:
        serial.append(await query_inference_server(p))
    t_serial = time.perf_counter() - t0
    print(f"Serial:  {t_serial:.2f} s")

    # Async (all concurrent)
    t0 = time.perf_counter()
    results = await batch_inference(prompts)
    t_async = time.perf_counter() - t0
    print(f"Async:   {t_async:.2f} s")
    print(f"Speedup: {t_serial / t_async:.0f}×")
    print(f"\nAll done in 1 thread, ~{len(prompts)} KB memory.")


if __name__ == "__main__":
    asyncio.run(main())