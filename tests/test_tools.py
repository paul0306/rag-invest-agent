# Tool-level tests.
from datetime import UTC, datetime

import httpx

from app.services.news_service import NewsArticle, search_news
from app.services.tools import risk_analyzer


def test_search_news_returns_structured_summary(monkeypatch) -> None:
    def fake_fetch(_plan):
        return [
            NewsArticle(
                title="NVIDIA raises guidance after strong AI server demand",
                source="Reuters",
                published_at=datetime(2026, 4, 10, tzinfo=UTC),
                url="https://example.com/nvda-1",
                summary="AI demand stays strong.",
            ),
            NewsArticle(
                title="NVIDIA faces new export restriction review",
                source="Bloomberg",
                published_at=datetime(2026, 4, 9, tzinfo=UTC),
                url="https://example.com/nvda-2",
                summary="Regulatory pressure remains in focus.",
            ),
        ]

    monkeypatch.setattr("app.services.news_service._fetch_news_articles", fake_fetch)
    result = search_news("Analyze NVIDIA investment risk over the next 6 months.")

    assert "[news_window=recent | subject=NVIDIA | articles=2]" in result
    assert "Bullish signals:" in result
    assert "Bearish signals:" in result
    assert "Sources:" in result
    assert "Reuters | 2026-04-10 | https://example.com/nvda-1" in result


def test_search_news_falls_back_when_feed_is_unavailable(monkeypatch) -> None:
    def fake_fetch(_plan):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr("app.services.news_service._fetch_news_articles", fake_fetch)
    result = search_news("nvidia")

    assert "Live news lookup is temporarily unavailable." in result
    assert "articles=0" in result


def test_risk_analyzer_mentions_risk() -> None:
    result = risk_analyzer.invoke("nvidia")
    assert "risk" in result.lower() or "valuation" in result.lower()
