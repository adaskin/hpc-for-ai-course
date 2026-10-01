"""
Week 4 Part III – Agent swarm: concurrent tool calls.
Run: python 07_agent_swarm.py

10 agents, each queries 3 tools concurrently. 30 async calls, 1 thread.
"""
import asyncio
import random
import time


async def call_tool(agent_id: int, tool: str) -> str:
    """Simulate an agent calling a tool (search, calculator, code_exec)."""
    latency = random.uniform(0.1, 0.4)
    await asyncio.sleep(latency)
    return f"Agent {agent_id} ← {tool}: result (took {latency:.2f}s)"


async def agent_loop(agent_id: int) -> list[str]:
    """One agent queries all its tools concurrently."""
    tools = ["web_search", "calculator", "code_executor"]
    tasks = [call_tool(agent_id, t) for t in tools]
    return await asyncio.gather(*tasks)


async def swarm():
    n_agents = 10
    print(f"Launching {n_agents} agents, 3 tools each ({n_agents * 3} concurrent calls)\n")

    t0 = time.perf_counter()
    agents = [agent_loop(i) for i in range(n_agents)]
    all_results = await asyncio.gather(*agents)
    elapsed = time.perf_counter() - t0

    for results in all_results:
        for r in results:
            print(f"  {r}")

    print(f"\nTotal: {elapsed:.2f} s (1 thread, {n_agents * 3} concurrent calls)")
    print(f"Serial would be: ~{n_agents * 3 * 0.25:.1f} s")


if __name__ == "__main__":
    asyncio.run(swarm())