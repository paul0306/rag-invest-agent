# Tool-level tests.
from app.services.news_service import search_news
from app.services.tools import risk_analyzer


def test_search_news_returns_text() -> None:
    result = search_news("nvidia")
    assert "-" in result


def test_risk_analyzer_mentions_risk() -> None:
    result = risk_analyzer.invoke("nvidia")
    assert "risk" in result.lower() or "valuation" in result.lower()
