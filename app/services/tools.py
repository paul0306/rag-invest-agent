from __future__ import annotations

from langchain.tools import tool

from app.services.news_service import search_news
from app.services.rag_service import retrieve_docs
import logging


logger = logging.getLogger(__name__)

@tool
def rag_search(query: str) -> str:
    """Search local financial research documents using hybrid RAG retrieval."""
    logger.info(f"rag_search called with query={query}")
    docs = retrieve_docs(query)
    return "\n".join([doc.page_content for doc in docs])


@tool
def news_search(query: str) -> str:
    """Retrieve recent market headlines or a mock news summary for a company."""
    logger.info(f"news_search called with query={query}")
    return search_news(query)


@tool
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
