"""
Week 4 Part III – Async streaming: token-by-token LLM generation.
Run: python 08_async_streaming.py

Simulates streaming tokens from an LLM (like OpenAI stream=True).
"""
import asyncio


async def llm_stream(prompt: str):
    """Simulate token-by-token generation from an LLM."""
    response = (
        f"Backpropagation computes gradients of the loss function "
        f"with respect to model parameters using the chain rule, "
        f"enabling efficient training of deep neural networks."
    )
    tokens = response.split()
    for token in tokens:
        await asyncio.sleep(0.05)  # generation latency per token
        yield token


async def consume_stream(prompt: str):
    print(f"Prompt: {prompt}")
    print(f"Response: ", end="", flush=True)
    count = 0
    async for token in llm_stream(prompt):
        print(token, end=" ", flush=True)
        count += 1
    print(f"\n\n[{count} tokens streamed, ~{count * 0.05:.1f}s total]")


async def multiple_streams():
    """Stream from 3 'models' concurrently."""
    prompts = [
        "What is backpropagation?",
        "What is attention?",
        "What is gradient descent?",
    ]
    tasks = [consume_stream(p) for p in prompts]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    print("=== Single stream ===\n")
    asyncio.run(consume_stream("What is backpropagation?"))

    print("\n=== 3 concurrent streams ===\n")
    asyncio.run(multiple_streams())