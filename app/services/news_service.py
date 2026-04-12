from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import html
import logging
import re
from typing import Iterable
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import httpx

from app.utils.config import get_settings


logger = logging.getLogger(__name__)


COMPANY_ALIASES = {
    "nvda": "NVIDIA",
    "nvidia": "NVIDIA",
    "amd": "AMD",
    "tsla": "Tesla",
    "tesla": "Tesla",
    "aapl": "Apple",
    "apple": "Apple",
    "msft": "Microsoft",
    "microsoft": "Microsoft",
    "amzn": "Amazon",
    "amazon": "Amazon",
    "goog": "Alphabet",
    "googl": "Alphabet",
    "alphabet": "Alphabet",
    "meta": "Meta",
    "netflix": "Netflix",
    "nflx": "Netflix",
    "tsm": "TSMC",
    "tsmc": "TSMC",
}

BULLISH_KEYWORDS = {
    "beat",
    "beats",
    "surge",
    "growth",
    "record",
    "expansion",
    "partnership",
    "upgrade",
    "strong demand",
    "profit jump",
    "launch",
    "raises guidance",
}

BEARISH_KEYWORDS = {
    "miss",
    "cut",
    "downturn",
    "weak",
    "lawsuit",
    "probe",
    "investigation",
    "tariff",
    "ban",
    "restriction",
    "delay",
    "slump",
    "recall",
    "downgrade",
}

WATCHLIST_KEYWORDS = {
    "earnings",
    "guidance",
    "forecast",
    "regulation",
    "export",
    "pricing",
    "competition",
    "margin",
    "demand",
    "capex",
    "supply chain",
}


@dataclass(frozen=True)
class SearchPlan:
    company: str | None
    search_terms: list[str]
    market_focus: bool


@dataclass(frozen=True)
class NewsArticle:
    title: str
    source: str
    published_at: datetime | None
    url: str
    summary: str


def _normalize_company(query: str) -> str | None:
    normalized = query.lower()
    for alias, company in COMPANY_ALIASES.items():
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        if re.search(pattern, normalized):
            return company
    return None


def _build_search_plan(query: str) -> SearchPlan:
    company = _normalize_company(query)
    search_terms: list[str] = []

    if company:
        search_terms.append(f'"{company}"')
        search_terms.extend(["stock", "earnings", "guidance"])

    normalized = query.lower()
    if any(word in normalized for word in ("risk", "downside", "headwind", "exposure")):
        search_terms.extend(["regulation", "competition", "margin"])
    elif any(word in normalized for word in ("growth", "bull", "upside")):
        search_terms.extend(["demand", "partnership", "expansion"])

    if not search_terms:
        search_terms.append(f'"{query.strip()}"')

    return SearchPlan(company=company, search_terms=search_terms, market_focus=company is not None)


def _build_google_news_url(plan: SearchPlan) -> str:
    settings = get_settings()
    search_query = " ".join(plan.search_terms)
    params = {
        "q": search_query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    return f"{settings.news_base_url}?{urlencode(params)}"


def _parse_pub_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(html.unescape(value).split())


def _parse_rss(xml_text: str) -> list[NewsArticle]:
    root = ET.fromstring(xml_text)
    items = root.findall(".//item")
    articles: list[NewsArticle] = []

    for item in items:
        title = _clean_text(item.findtext("title"))
        url = _clean_text(item.findtext("link"))
        source = _clean_text(item.findtext("source")) or "Unknown source"
        summary = _clean_text(item.findtext("description"))
        if not title or not url:
            continue
        articles.append(
            NewsArticle(
                title=title,
                source=source,
                published_at=_parse_pub_date(item.findtext("pubDate")),
                url=url,
                summary=summary,
            )
        )

    return articles


def _fetch_news_articles(plan: SearchPlan) -> list[NewsArticle]:
    settings = get_settings()
    url = _build_google_news_url(plan)
    headers = {"User-Agent": "rag-invest-agent/1.0"}
    logger.info("Fetching news feed: %s", url)

    with httpx.Client(timeout=settings.news_timeout_seconds, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        articles = _parse_rss(response.text)

    return _dedupe_articles(articles)[: settings.news_max_results]


def _dedupe_articles(articles: Iterable[NewsArticle]) -> list[NewsArticle]:
    deduped: list[NewsArticle] = []
    seen: set[tuple[str, str]] = set()

    for article in articles:
        key = (article.title.casefold(), article.source.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(article)

    return deduped


def _classify_articles(articles: Iterable[NewsArticle]) -> tuple[list[str], list[str], list[str]]:
    bullish: list[str] = []
    bearish: list[str] = []
    watch_items: list[str] = []

    for article in articles:
        headline = article.title.lower()
        if any(keyword in headline for keyword in BULLISH_KEYWORDS):
            bullish.append(article.title)
        if any(keyword in headline for keyword in BEARISH_KEYWORDS):
            bearish.append(article.title)
        if any(keyword in headline for keyword in WATCHLIST_KEYWORDS):
            watch_items.append(article.title)

    return bullish[:3], bearish[:3], watch_items[:3]


def _format_date(value: datetime | None) -> str:
    if value is None:
        return "date unavailable"
    return value.astimezone(UTC).strftime("%Y-%m-%d")


def _format_news_summary(query: str, plan: SearchPlan, articles: list[NewsArticle]) -> str:
    if not articles:
        subject = plan.company or query.strip()
        return "\n".join(
            [
                f"[news_window=recent | subject={subject or 'market'} | articles=0]",
                "Major developments:",
                "- No recent market headlines were returned for this query.",
                "Watch items:",
                "- Try a more specific company name or ticker.",
            ]
        )

    bullish, bearish, watch_items = _classify_articles(articles)
    subject = plan.company or query.strip() or "market"
    lines = [f"[news_window=recent | subject={subject} | articles={len(articles)}]", "Major developments:"]
    lines.extend(f"- {article.title}" for article in articles[:3])

    lines.append("Bullish signals:")
    lines.extend(f"- {item}" for item in bullish or ["No clearly bullish headline signal detected in the latest sample."])

    lines.append("Bearish signals:")
    lines.extend(f"- {item}" for item in bearish or ["No clearly bearish headline signal detected in the latest sample."])

    lines.append("Watch items:")
    default_watch = "Monitor earnings guidance, regulation, competition, and demand updates as new articles arrive."
    lines.extend(f"- {item}" for item in watch_items or [default_watch])

    lines.append("Sources:")
    for article in articles:
        lines.append(f"- {article.source} | {_format_date(article.published_at)} | {article.url}")

    return "\n".join(lines)


def search_news(query: str) -> str:
    plan = _build_search_plan(query)

    try:
        articles = _fetch_news_articles(plan)
    except (httpx.HTTPError, ET.ParseError) as exc:
        logger.warning("Live news lookup failed for query=%s: %s", query, exc)
        subject = plan.company or query.strip() or "market"
        return "\n".join(
            [
                f"[news_window=recent | subject={subject} | articles=0]",
                "Major developments:",
                "- Live news lookup is temporarily unavailable.",
                "Watch items:",
                "- Check network access, RSS availability, or try again later.",
            ]
        )

    return _format_news_summary(query, plan, articles)
