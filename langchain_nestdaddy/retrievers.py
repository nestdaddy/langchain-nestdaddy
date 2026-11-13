"""NestDaddy news retriever: real-time news as LangChain Documents, optionally with the full article text."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator

from langchain_nestdaddy._client import NestDaddyAPIError, NestDaddyAPIWrapper

META_FIELDS = ("title", "url", "source_domain", "published_timestamp", "country")


class NestDaddyNewsRetriever(BaseRetriever):
    """Retrieve real-time news from around the world for a query.

    Each article becomes a ``Document``: the summary (or, with ``full_text=True``, the article text) is the
    ``page_content``; title, url, source, publish time and country are in ``metadata``.

    Setup:
        .. code-block:: bash

            pip install -U langchain-nestdaddy
            export NESTDADDY_API_KEY="your-key"

    Instantiate:
        .. code-block:: python

            from langchain_nestdaddy import NestDaddyNewsRetriever

            retriever = NestDaddyNewsRetriever(k=5, days=7, full_text=True)

    Usage:
        .. code-block:: python

            docs = retriever.invoke("chip export rules")
    """

    api_wrapper: NestDaddyAPIWrapper
    k: int = Field(default=5, ge=1, le=50)
    """How many articles to return."""
    days: int = Field(default=30, ge=1, le=120)
    """Only news from the last N days."""
    country: Optional[str] = None
    """Optional two-letter country code, e.g. US, GB, IN, SG, ID."""
    full_text: bool = False
    """Read each article in full (one extra API call per article) instead of using its summary."""
    max_chars: int = Field(default=6000, ge=500, le=30000)
    """Characters kept from each full article."""

    @model_validator(mode="before")
    @classmethod
    def _build_wrapper(cls, values: Any) -> Any:
        if isinstance(values, dict) and "api_wrapper" not in values:
            kwargs = {k: values.pop(k) for k in ("api_key", "base_url") if k in values}
            values["api_wrapper"] = NestDaddyAPIWrapper(**kwargs)
        return values

    # ------------------------------------------------------------------ helpers
    def _params(self, query: str) -> Dict[str, Any]:
        return {"q": query, "days": self.days, "limit": self.k,
                "country": self.country.upper() if self.country else None}

    @staticmethod
    def _to_doc(item: Dict[str, Any], text: Optional[str] = None) -> Document:
        content = text or item.get("summary") or item.get("description") or item.get("title") or ""
        meta = {f: item[f] for f in META_FIELDS if item.get(f) not in (None, "")}
        meta["full_text"] = bool(text)
        return Document(page_content=content, metadata=meta)

    def _items(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [r for r in data.get("results") or [] if isinstance(r, dict) and r.get("url")][: self.k]

    def _read(self, url: str) -> Optional[str]:
        try:
            page = self.api_wrapper.get("/api/v1/extract", {"url": url, "max_chars": self.max_chars})
        except NestDaddyAPIError:
            return None          # keep the summary when one article cannot be read
        text = page.get("text") or ""
        return text if len(text) > 200 else None

    async def _aread(self, url: str) -> Optional[str]:
        try:
            page = await self.api_wrapper.aget("/api/v1/extract", {"url": url, "max_chars": self.max_chars})
        except NestDaddyAPIError:
            return None
        text = page.get("text") or ""
        return text if len(text) > 200 else None

    # ------------------------------------------------------------------ BaseRetriever
    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        items = self._items(self.api_wrapper.get("/api/v1/news/search", self._params(query)))
        texts = [self._read(i["url"]) if self.full_text else None for i in items]
        return [self._to_doc(i, t) for i, t in zip(items, texts)]

    async def _aget_relevant_documents(self, query: str, *,
                                       run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        items = self._items(await self.api_wrapper.aget("/api/v1/news/search", self._params(query)))
        if self.full_text:
            texts = await asyncio.gather(*(self._aread(i["url"]) for i in items))
        else:
            texts = [None] * len(items)
        return [self._to_doc(i, t) for i, t in zip(items, texts)]
