"""Behaviour tests with a fake NestDaddy API (httpx.MockTransport), sync and async."""

import json
from typing import Any, Dict, List

import httpx
import pytest

from langchain_nestdaddy import (
    NestDaddyAPIError,
    NestDaddyAPIWrapper,
    NestDaddyNewsRetriever,
    NestDaddyNewsSearch,
    NestDaddyReadPage,
    NestDaddySearch,
)

LONG = "Full article text. " * 40


def fake_api(seen: List[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("X-API-Key") != "test-key":
            return httpx.Response(401, json={"error": "Invalid API key", "message": "The key is not valid"})
        path, q = request.url.path, dict(request.url.params)
        if path == "/api/v1/search":
            return httpx.Response(200, json={"query": q["q"], "results": [
                {"title": "MCP spec", "url": "https://modelcontextprotocol.io", "snippet": "x" * 900,
                 "domain": "modelcontextprotocol.io", "type": "web", "internal_score": 9}]})
        if path == "/api/v1/news/search":
            return httpx.Response(200, json={"query": q["q"], "total": 2, "results": [
                {"title": "Fed holds rates", "url": "https://news.example/a", "source_domain": "news.example",
                 "published_timestamp": "2026-11-13T10:00:00Z", "summary": "The Fed held rates.", "country": "US"},
                {"title": "Markets rise", "url": "https://news.example/b", "source_domain": "news.example",
                 "published_timestamp": "2026-11-13T09:00:00Z", "summary": "Stocks rose.", "country": "US"}]})
        if path == "/api/v1/extract":
            if q["url"].endswith("/b"):
                return httpx.Response(422, json={"error": "Unreadable", "message": "No article text"})
            return httpx.Response(200, json={"title": "Fed holds rates", "final_url": q["url"],
                                             "text": LONG, "length": len(LONG), "truncated": False})
        return httpx.Response(404, json={"error": "Not found", "message": path})
    return httpx.MockTransport(handler)


def wrapper(seen: List[httpx.Request], key: str = "test-key") -> NestDaddyAPIWrapper:
    return NestDaddyAPIWrapper(api_key=key, transport=fake_api(seen))


def test_missing_key_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NESTDADDY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NESTDADDY_API_KEY"):
        NestDaddySearch()


def test_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NESTDADDY_API_KEY", "env-key")
    assert NestDaddySearch().api_wrapper.api_key.get_secret_value() == "env-key"


def test_key_is_not_printed() -> None:
    w = NestDaddyAPIWrapper(api_key="secret-value")
    assert "secret-value" not in repr(w) and "secret-value" not in str(w.model_dump())


def test_web_search_trims_and_picks_fields() -> None:
    seen: List[httpx.Request] = []
    out = NestDaddySearch(api_wrapper=wrapper(seen), max_results=3).invoke({"query": "mcp"})
    assert seen[0].url.params["limit"] == "3" and seen[0].url.params["q"] == "mcp"
    r = out["results"][0]
    assert set(r) == {"title", "url", "snippet", "domain", "type"}   # internal fields dropped
    assert len(r["snippet"]) <= 403


def test_news_search_params_and_output() -> None:
    seen: List[httpx.Request] = []
    out = NestDaddyNewsSearch(api_wrapper=wrapper(seen), days=7).invoke(
        {"query": "fed", "country": "us", "max_results": 2})
    p = seen[0].url.params
    assert (p["country"], p["days"], p["limit"]) == ("US", "7", "2")
    assert out["count"] == 2 and out["total"] == 2 and out["results"][0]["source_domain"] == "news.example"


def test_optional_params_are_not_sent() -> None:
    seen: List[httpx.Request] = []
    NestDaddyNewsSearch(api_wrapper=wrapper(seen)).invoke({"query": "fed"})
    assert "country" not in seen[0].url.params


def test_read_page() -> None:
    seen: List[httpx.Request] = []
    out = NestDaddyReadPage(api_wrapper=wrapper(seen), max_chars=2000).invoke({"url": "https://news.example/a"})
    assert seen[0].url.params["max_chars"] == "2000" and out["text"].startswith("Full article text")


def test_api_error_is_raised_with_hint() -> None:
    seen: List[httpx.Request] = []
    with pytest.raises(NestDaddyAPIError, match="dashboard") as e:
        wrapper(seen, key="wrong").get("/api/v1/search", {"q": "x"})
    assert e.value.status_code == 401


def test_tool_as_agent_tool_message() -> None:
    """Tools called by an agent return a ToolMessage whose content is the JSON result."""
    seen: List[httpx.Request] = []
    tool = NestDaddyNewsSearch(api_wrapper=wrapper(seen))
    msg = tool.invoke({"type": "tool_call", "id": "call_1", "name": tool.name, "args": {"query": "fed"}})
    assert msg.tool_call_id == "call_1" and "Fed holds rates" in msg.content


def test_retriever_summaries() -> None:
    seen: List[httpx.Request] = []
    docs = NestDaddyNewsRetriever(api_wrapper=wrapper(seen), k=2).invoke("fed")
    assert [d.page_content for d in docs] == ["The Fed held rates.", "Stocks rose."]
    assert docs[0].metadata["url"] == "https://news.example/a" and docs[0].metadata["full_text"] is False
    assert len(seen) == 1


def test_retriever_full_text_keeps_summary_when_unreadable() -> None:
    seen: List[httpx.Request] = []
    docs = NestDaddyNewsRetriever(api_wrapper=wrapper(seen), k=2, full_text=True).invoke("fed")
    assert docs[0].page_content.startswith("Full article text") and docs[0].metadata["full_text"] is True
    assert docs[1].page_content == "Stocks rose." and docs[1].metadata["full_text"] is False
    assert len(seen) == 3


async def test_async_tool_and_retriever() -> None:
    seen: List[httpx.Request] = []
    out = await NestDaddySearch(api_wrapper=wrapper(seen)).ainvoke({"query": "mcp"})
    assert out["count"] == 1
    docs = await NestDaddyNewsRetriever(api_wrapper=wrapper(seen), k=2, full_text=True).ainvoke("fed")
    assert docs[0].metadata["full_text"] is True and len(docs) == 2


def test_output_is_json_serialisable() -> None:
    seen: List[httpx.Request] = []
    out: Dict[str, Any] = NestDaddyNewsSearch(api_wrapper=wrapper(seen)).invoke({"query": "fed"})
    json.dumps(out)
