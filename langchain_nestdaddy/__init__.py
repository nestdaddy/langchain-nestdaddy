"""LangChain integration for NestDaddy: give your AI today's internet.

Live web search, real-time breaking news from around the world and full page reading, from NestDaddy's own
index. Get a key at https://nestdaddy.com/mcp and set ``NESTDADDY_API_KEY``.
"""

from importlib import metadata

from langchain_nestdaddy._client import NestDaddyAPIError, NestDaddyAPIWrapper
from langchain_nestdaddy.retrievers import NestDaddyNewsRetriever
from langchain_nestdaddy.tools import NestDaddyNewsSearch, NestDaddyReadPage, NestDaddySearch

try:
    __version__ = metadata.version(__package__ or "langchain_nestdaddy")
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = [
    "NestDaddyAPIError",
    "NestDaddyAPIWrapper",
    "NestDaddyNewsRetriever",
    "NestDaddyNewsSearch",
    "NestDaddyReadPage",
    "NestDaddySearch",
    "__version__",
]
