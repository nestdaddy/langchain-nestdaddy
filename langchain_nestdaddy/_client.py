"""Small HTTP client for the NestDaddy REST API, shared by the tools and the retriever."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

DEFAULT_BASE_URL = "https://nestdaddy.com"
USER_AGENT = "langchain-nestdaddy/0.1.1"


class NestDaddyAPIError(RuntimeError):
    """Raised when the NestDaddy API answers with an error (bad key, plan limit, bad input)."""

    def __init__(self, status_code: int, error: str, message: str) -> None:
        self.status_code = status_code
        self.error = error
        hint = ""
        if status_code in (401, 403):
            hint = " Check your key at https://nestdaddy.com/dashboard"
        elif status_code == 429:
            hint = " You reached your plan's limit: https://nestdaddy.com/mcp"
        super().__init__(f"NestDaddy API error {status_code} ({error}): {message}.{hint}")


class NestDaddyAPIWrapper(BaseModel):
    """Calls the NestDaddy API with your key.

    The key is read from ``api_key`` or the ``NESTDADDY_API_KEY`` environment variable.
    Get one at https://nestdaddy.com/mcp (Free plan: 1,000 calls a month).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    api_key: Optional[SecretStr] = Field(default=None)
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0
    transport: Optional[Any] = Field(default=None, exclude=True, repr=False)
    """Optional httpx transport (used by tests). Leave unset in normal use."""

    @model_validator(mode="after")
    def _resolve_key(self) -> "NestDaddyAPIWrapper":
        if self.api_key is None:
            env = os.environ.get("NESTDADDY_API_KEY", "").strip()
            if env:
                self.api_key = SecretStr(env)
        if self.api_key is None or not self.api_key.get_secret_value():
            raise ValueError(
                "NestDaddy API key not found. Pass api_key=... or set NESTDADDY_API_KEY. "
                "Get a key at https://nestdaddy.com/mcp"
            )
        return self

    # ------------------------------------------------------------------ internals
    def _headers(self) -> Dict[str, str]:
        assert self.api_key is not None
        return {
            "X-API-Key": self.api_key.get_secret_value(),
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }

    @staticmethod
    def _clean(params: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in params.items() if v not in (None, "")}

    @staticmethod
    def _parse(response: httpx.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            raise NestDaddyAPIError(response.status_code, "invalid_response",
                                    "the API did not return JSON") from None
        if response.status_code != 200:
            raise NestDaddyAPIError(response.status_code, str(data.get("error") or "error"),
                                    str(data.get("message") or data.get("error") or response.reason_phrase))
        return data

    def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self.transport) as client:
            return self._parse(client.get(path, params=self._clean(params), headers=self._headers()))

    async def aget(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                     transport=self.transport) as client:
            return self._parse(await client.get(path, params=self._clean(params), headers=self._headers()))
