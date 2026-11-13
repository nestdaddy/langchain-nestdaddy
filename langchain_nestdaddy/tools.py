"""NestDaddy tools for LangChain agents: web search, real-time news search and full page reading."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from langchain_nestdaddy._client import NestDaddyAPIWrapper

SNIPPET_CHARS = 400
NEWS_FIELDS = ("title", "url", "source_domain", "published_timestamp", "summary", "description", "country")
WEB_FIELDS = ("title", "url", "snippet", "domain", "type")


def _pick(item: Dict[str, Any], fields: Tuple[str, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in fields:
        v = item.get(f)
        if v in (None, "", [], {}):
            continue
        if isinstance(v, str) and len(v) > SNIPPET_CHARS:
            v = v[:SNIPPET_CHARS].rstrip() + "..."
        out[f] = v
    return out


class _NestDaddyTool(BaseTool):
    """Shared setup: every tool owns an API wrapper, built from ``api_key`` or ``NESTDADDY_API_KEY``."""

    api_wrapper: NestDaddyAPIWrapper

    @model_validator(mode="before")
    @classmethod
    def _build_wrapper(cls, values: Any) -> Any:
        if isinstance(values, dict) and "api_wrapper" not in values:
            kwargs = {k: values.pop(k) for k in ("api_key", "base_url") if k in values}
            values["api_wrapper"] = NestDaddyAPIWrapper(**kwargs)
        return values


# ---------------------------------------------------------------------------------------------- web search
class NestDaddySearchInput(BaseModel):
    query: str = Field(description="Short key words to search the web for, not a whole sentence.")
    max_results: Optional[int] = Field(default=None, ge=1, le=50,
                                       description="How many results to return (1-50).")


class NestDaddySearch(_NestDaddyTool):
    """Search the web with NestDaddy's own index.

    Setup:
        .. code-block:: bash

            pip install -U langchain-nestdaddy
            export NESTDADDY_API_KEY="your-key"

    Instantiate:
        .. code-block:: python

            from langchain_nestdaddy import NestDaddySearch

            tool = NestDaddySearch(max_results=5)

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "model context protocol"})
    """

    name: str = "nestdaddy_search"
    description: str = (
        "Search the web with NestDaddy's own index. Returns the title, url, snippet and domain of each page. "
        "For time-sensitive questions fresh news articles are mixed in. Use short key words. "
        "Call nestdaddy_read_page on a url when the snippet is not enough."
    )
    args_schema: Type[BaseModel] = NestDaddySearchInput
    max_results: int = 10

    def _params(self, query: str, max_results: Optional[int]) -> Dict[str, Any]:
        return {"q": query, "limit": max_results or self.max_results}

    @staticmethod
    def _format(data: Dict[str, Any]) -> Dict[str, Any]:
        results = [_pick(r, WEB_FIELDS) for r in data.get("results") or [] if isinstance(r, dict)]
        return {"query": data.get("query"), "count": len(results), "results": results}

    def _run(self, query: str, max_results: Optional[int] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        return self._format(self.api_wrapper.get("/api/v1/search", self._params(query, max_results)))

    async def _arun(self, query: str, max_results: Optional[int] = None,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> Dict[str, Any]:
        return self._format(await self.api_wrapper.aget("/api/v1/search", self._params(query, max_results)))


# ---------------------------------------------------------------------------------------------- news search
class NestDaddyNewsSearchInput(BaseModel):
    query: str = Field(description="Key words a headline would use, in the local language for local news.")
    country: Optional[str] = Field(default=None, min_length=2, max_length=2,
                                   description="Optional two-letter country code, e.g. US, GB, IN, SG, ID.")
    days: Optional[int] = Field(default=None, ge=1, le=120, description="Only news from the last N days (1-120).")
    max_results: Optional[int] = Field(default=None, ge=1, le=50,
                                       description="How many articles to return (1-50).")


class NestDaddyNewsSearch(_NestDaddyTool):
    """Search real-time news from around the world by key words, newest and best match first.

    Setup:
        .. code-block:: bash

            pip install -U langchain-nestdaddy
            export NESTDADDY_API_KEY="your-key"

    Instantiate:
        .. code-block:: python

            from langchain_nestdaddy import NestDaddyNewsSearch

            tool = NestDaddyNewsSearch(days=7, max_results=5)

    Invoke:
        .. code-block:: python

            tool.invoke({"query": "interest rates", "country": "US"})
    """

    name: str = "nestdaddy_news_search"
    description: str = (
        "Search real-time breaking news from around the world by key words: title, url, source, publish time and "
        "summary of each article, newest and best match first. The best tool for questions about recent events, "
        "people, companies and places. Optional country (two-letter code) and days (1-120)."
    )
    args_schema: Type[BaseModel] = NestDaddyNewsSearchInput
    max_results: int = 10
    days: int = 30

    def _params(self, query: str, country: Optional[str], days: Optional[int],
                max_results: Optional[int]) -> Dict[str, Any]:
        return {"q": query, "country": country.upper() if country else None, "days": days or self.days,
                "limit": max_results or self.max_results}

    @staticmethod
    def _format(data: Dict[str, Any]) -> Dict[str, Any]:
        results = [_pick(r, NEWS_FIELDS) for r in data.get("results") or [] if isinstance(r, dict)]
        return {"query": data.get("query"), "count": len(results), "total": data.get("total"), "results": results}

    def _run(self, query: str, country: Optional[str] = None, days: Optional[int] = None,
             max_results: Optional[int] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        return self._format(self.api_wrapper.get("/api/v1/news/search",
                                                 self._params(query, country, days, max_results)))

    async def _arun(self, query: str, country: Optional[str] = None, days: Optional[int] = None,
                    max_results: Optional[int] = None,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> Dict[str, Any]:
        return self._format(await self.api_wrapper.aget("/api/v1/news/search",
                                                        self._params(query, country, days, max_results)))


# ---------------------------------------------------------------------------------------------- read page
class NestDaddyReadPageInput(BaseModel):
    url: str = Field(description="The full http(s) url of the page or article to read.")
    max_chars: Optional[int] = Field(default=None, ge=500, le=30000,
                                     description="Maximum characters of text to return (500-30000).")


class NestDaddyReadPage(_NestDaddyTool):
    """Read the clean main text of any web page or news article (no menus, no ads).

    Setup:
        .. code-block:: bash

            pip install -U langchain-nestdaddy
            export NESTDADDY_API_KEY="your-key"

    Instantiate:
        .. code-block:: python

            from langchain_nestdaddy import NestDaddyReadPage

            tool = NestDaddyReadPage(max_chars=8000)

    Invoke:
        .. code-block:: python

            tool.invoke({"url": "https://en.wikipedia.org/wiki/Model_Context_Protocol"})
    """

    name: str = "nestdaddy_read_page"
    description: str = (
        "Read the full readable text of one web page or news article: title, author, date and main text, "
        "without menus or ads. Use it after a search when you need the details behind a result."
    )
    args_schema: Type[BaseModel] = NestDaddyReadPageInput
    max_chars: int = 8000

    @staticmethod
    def _format(data: Dict[str, Any]) -> Dict[str, Any]:
        keep = ("title", "final_url", "author", "published", "text", "truncated", "length")
        return {k: data[k] for k in keep if data.get(k) not in (None, "")}

    def _run(self, url: str, max_chars: Optional[int] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        return self._format(self.api_wrapper.get("/api/v1/extract",
                                                 {"url": url, "max_chars": max_chars or self.max_chars}))

    async def _arun(self, url: str, max_chars: Optional[int] = None,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> Dict[str, Any]:
        return self._format(await self.api_wrapper.aget("/api/v1/extract",
                                                        {"url": url, "max_chars": max_chars or self.max_chars}))


__all__: List[str] = [
    "NestDaddySearch", "NestDaddySearchInput",
    "NestDaddyNewsSearch", "NestDaddyNewsSearchInput",
    "NestDaddyReadPage", "NestDaddyReadPageInput",
]
