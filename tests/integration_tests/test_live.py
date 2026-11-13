"""Live tests against the real NestDaddy API. Run with: NESTDADDY_API_KEY=... pytest -m integration"""

import os

import pytest

from langchain_nestdaddy import NestDaddyNewsRetriever, NestDaddyNewsSearch, NestDaddyReadPage, NestDaddySearch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("NESTDADDY_API_KEY"), reason="NESTDADDY_API_KEY not set"),
]


def test_live_web_search() -> None:
    out = NestDaddySearch(max_results=3).invoke({"query": "model context protocol"})
    assert out["count"] >= 1 and out["results"][0]["url"].startswith("http")


def test_live_news_search() -> None:
    out = NestDaddyNewsSearch(max_results=3, days=7).invoke({"query": "AI"})
    assert out["count"] >= 1 and out["results"][0]["title"]


def test_live_read_page() -> None:
    out = NestDaddyReadPage(max_chars=1000).invoke({"url": "https://en.wikipedia.org/wiki/Web_search_engine"})
    assert len(out.get("text", "")) > 200


def test_live_retriever() -> None:
    docs = NestDaddyNewsRetriever(k=2, days=7).invoke("AI")
    assert docs and docs[0].metadata["url"].startswith("http")
