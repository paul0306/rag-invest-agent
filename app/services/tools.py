# LangChain tools exposed to the agent.
from __future__ import annotations

from langchain.tools import tool

from app.services.news_service import search_news
from app.services.rag_service import hybrid_retrieve
import logging


# Module-level logger keeps tool calls visible during local debugging.
logger = logging.getLogger(__name__)

@tool
# Retrieval tool: return local document context plus cache metadata.
def rag_search(query: str) -> str:
    """Search local financial research documents using hybrid RAG retrieval."""
    logger.info(f"rag_search called with query={query}")
    bundle = hybrid_retrieve(query)
    cache_line = f"[retrieval_cache_hit={str(bundle.cache_hit).lower()} | strategy={bundle.strategy}]"
    return f"{cache_line}\n{bundle.context}"


@tool
# News tool: return mock bullet headlines for recent-company context.
def news_search(query: str) -> str:
    """Retrieve recent market headlines or a mock news summary for a company."""
    logger.info(f"news_search called with query={query}")
    return search_news(query)


@tool
# Heuristic risk tool: add a few domain-specific downside checks.
def risk_analyzer(query: str) -> str:
    """Generate a lightweight heuristic risk checklist for an equity-research question."""
    normalized = query.lower()
    risks = [
        "Valuation multiple compression risk if growth expectations cool.",
        "Macro or capex slowdown risk from cloud and enterprise customers.",
        "Geopolitical and export-control risk across semiconductor supply chains.",
    ]
    if "amd" in normalized:
        risks.append("Execution risk in ramping competitive AI accelerator products.")
    if "nvidia" in normalized or "nvda" in normalized:
        risks.append("Customer concentration risk among large cloud providers.")
    return "\n".join(f"- {risk}" for risk in risks)
