from __future__ import annotations

import statistics
import time

from app.services.rag_service import hybrid_retrieve

QUERIES = [
    "Analyze NVIDIA investment risk from recent earnings and AI demand.",
    "What are the biggest margin and competition risks for NVIDIA?",
    "Summarize bullish factors from semiconductor demand trends.",
]


def run_once(query: str) -> float:
    started = time.perf_counter()
    _ = hybrid_retrieve(query)
    return time.perf_counter() - started


if __name__ == "__main__":
    latencies = [run_once(query) for query in QUERIES]
    print("Benchmark results")
    print("-----------------")
    for query, latency in zip(QUERIES, latencies, strict=True):
        print(f"{latency:.4f}s | {query}")
    print(f"avg={statistics.mean(latencies):.4f}s")
