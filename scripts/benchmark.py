# Lightweight benchmark script for retrieval and agent latency profiling.
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import statistics
import time
from typing import Any
from app.services.agent_service import run_analysis
from app.services.rag_service import hybrid_retrieve
from app.services.tools import rag_search


# Fixed benchmark prompts keep repeated latency runs comparable.
QUERIES = [
    "Analyze NVIDIA investment risk from recent earnings and AI demand.",
    "What are the biggest margin and competition risks for NVIDIA?",
    "Summarize bullish factors from semiconductor demand trends.",
]


# Measure only the hybrid retrieval step.
def measure_retrieval(query: str) -> tuple[float, Any]:
    started = time.perf_counter()
    docs = hybrid_retrieve(query)
    latency = time.perf_counter() - started
    return latency, docs


# Measure the retrieval tool wrapper used by the agent.
def measure_rag_tool(query: str) -> tuple[float, str]:
    started = time.perf_counter()
    result = rag_search.invoke({"query": query})
    latency = time.perf_counter() - started
    return latency, result


# Measure end-to-end latency for the full agent workflow.
def measure_agent(query: str) -> tuple[float, str]:
    started = time.perf_counter()
    result = run_analysis(query)
    latency = time.perf_counter() - started
    return latency, result


# Aggregate timing metrics for a single benchmark query.
def benchmark_query(query: str) -> dict[str, float]:
    retrieval_latency, docs = measure_retrieval(query)
    rag_tool_latency, _ = measure_rag_tool(query)
    agent_latency, _ = measure_agent(query)

    approximate_reasoning_latency = max(agent_latency - rag_tool_latency, 0.0)

    return {
        "retrieval_latency": retrieval_latency,
        "rag_tool_latency": rag_tool_latency,
        "agent_latency": agent_latency,
        "approx_reasoning_latency": approximate_reasoning_latency,
        "num_docs": float(len(docs.docs)),
    }


# Print averaged metrics and the warm-cache speedup estimate.
def print_summary(results: list[dict[str, float]]) -> None:
    retrievals = [r["retrieval_latency"] for r in results]
    rag_tools = [r["rag_tool_latency"] for r in results]
    agents = [r["agent_latency"] for r in results]
    reasonings = [r["approx_reasoning_latency"] for r in results]

    print("\nSummary")
    print("-------")
    print(f"avg retrieval latency:        {statistics.mean(retrievals):.4f}s")
    print(f"avg rag tool latency:         {statistics.mean(rag_tools):.4f}s")
    print(f"avg agent end-to-end latency: {statistics.mean(agents):.4f}s")
    print(f"avg approx reasoning latency: {statistics.mean(reasonings):.4f}s")

    if retrievals:
        cold = retrievals[0]
        warm = statistics.mean(retrievals[1:]) if len(retrievals) > 1 else retrievals[0]
        speedup = ((cold - warm) / cold * 100) if cold > 0 else 0.0

        print("\nCache effect")
        print("------------")
        print(f"cold retrieval latency: {cold:.4f}s")
        print(f"warm retrieval latency: {warm:.4f}s")
        print(f"retrieval speedup:      {speedup:.2f}%")


# Run the benchmark over the predefined query set.
def main() -> None:
    results: list[dict[str, float]] = []

    print("Benchmark results")
    print("-----------------")

    for query in QUERIES:
        metrics = benchmark_query(query)
        results.append(metrics)

        print(query)
        print(f"  retrieval:         {metrics['retrieval_latency']:.4f}s")
        print(f"  rag tool:          {metrics['rag_tool_latency']:.4f}s")
        print(f"  agent end-to-end:  {metrics['agent_latency']:.4f}s")
        print(f"  approx reasoning:  {metrics['approx_reasoning_latency']:.4f}s")
        print(f"  docs:              {int(metrics['num_docs'])}")
        print()

    print_summary(results)


if __name__ == "__main__":
    main()
