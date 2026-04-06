# Mock news adapter.
# The project can run with only a Gemini API key because recent headlines
# are represented by a small in-memory lookup table.
from __future__ import annotations

from typing import List

# Local fallback headlines keyed by ticker/company name.
MOCK_NEWS = {
    "nvda": [
        "Hyperscaler AI infrastructure spending remains elevated.",
        "GPU demand stays strong, but expectations are already high.",
        "Export restrictions and customer concentration remain watchpoints.",
    ],
    "nvidia": [
        "Hyperscaler AI infrastructure spending remains elevated.",
        "GPU demand stays strong, but expectations are already high.",
        "Export restrictions and customer concentration remain watchpoints.",
    ],
    "amd": [
        "Competitive pressure in AI accelerators continues to increase.",
        "Data center momentum improved, though execution remains critical.",
        "Margin expansion may depend on product-mix improvements.",
    ],
}


# Return bullet-form mock headlines so the agent can reason over "recent news".
def search_news(query: str) -> str:
    normalized = query.lower()
    hits: List[str] = []
    for key, items in MOCK_NEWS.items():
        if key in normalized:
            hits.extend(items)

    if not hits:
        hits = [
            "No company-specific live news source is configured yet.",
            "The app is using local mock headlines so the project stays runnable without extra API keys.",
            "Replace app/services/news_service.py with a real market-news adapter when you are ready.",
        ]

    return "\n".join(f"- {item}" for item in hits)
