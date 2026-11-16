# langchain-nestdaddy

**Give your AI today's internet.** Live web search, real-time breaking news from around the world and full page
reading for LangChain agents and RAG pipelines, from [NestDaddy](https://nestdaddy.com)'s own index.

[![PyPI](https://img.shields.io/pypi/v/langchain-nestdaddy)](https://pypi.org/project/langchain-nestdaddy/)
[![tests](https://github.com/nestdaddy/langchain-nestdaddy/actions/workflows/tests.yml/badge.svg)](https://github.com/nestdaddy/langchain-nestdaddy/actions/workflows/tests.yml)

```bash
pip install -U langchain-nestdaddy
export NESTDADDY_API_KEY="your-key"   # free key, 1,000 calls a month: https://nestdaddy.com/mcp
```

Step-by-step guide with agent and RAG examples:
[Give your LangChain agent live web search and real-time news](https://nestdaddy.com/blog/langchain-live-web-search-news).

| Class | What it does |
|---|---|
| `NestDaddySearch` | Web search on NestDaddy's own index (title, url, snippet, domain) |
| `NestDaddyNewsSearch` | Real-time news by key words, optional country and time window |
| `NestDaddyReadPage` | Clean main text of any page or article, no menus or ads |
| `NestDaddyNewsRetriever` | News as LangChain `Document`s for RAG, optionally with the full article text |

## Give an agent live search and news

```python
from langchain.agents import create_agent
from langchain_nestdaddy import NestDaddyNewsSearch, NestDaddyReadPage, NestDaddySearch

agent = create_agent(
    "anthropic:claude-sonnet-5-5",
    tools=[NestDaddySearch(max_results=5), NestDaddyNewsSearch(days=7), NestDaddyReadPage()],
    system_prompt="Use news search for anything recent, web search for everything else. "
                  "Read a page in full before you explain its details.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What did the Fed decide this week?"}]})
print(result["messages"][-1].content)
```

Any chat model that supports tool calling works the same way.

## Use the tools directly

```python
from langchain_nestdaddy import NestDaddyNewsSearch, NestDaddySearch, NestDaddyReadPage

NestDaddySearch(max_results=3).invoke({"query": "model context protocol"})
# {'query': 'model context protocol', 'count': 3, 'results': [{'title': ..., 'url': ..., 'snippet': ...}]}

NestDaddyNewsSearch().invoke({"query": "interest rates", "country": "US", "days": 7})
# {'query': 'interest rates', 'count': 10, 'total': ..., 'results': [{'title': ..., 'source_domain': ...,
#   'published_timestamp': ..., 'summary': ...}]}

NestDaddyReadPage(max_chars=8000).invoke({"url": "https://example.com/some-article"})
# {'title': ..., 'final_url': ..., 'text': ...}
```

Local news works in the local language: `NestDaddyNewsSearch().invoke({"query": "banjir Jakarta", "country": "ID"})`.

## RAG over today's news

```python
from langchain_nestdaddy import NestDaddyNewsRetriever

retriever = NestDaddyNewsRetriever(k=5, days=3, full_text=True)
docs = retriever.invoke("chip export rules")

for d in docs:
    print(d.metadata["title"], d.metadata["url"], d.metadata["published_timestamp"])
    print(d.page_content[:300])
```

With `full_text=True` each article is read in full (one extra API call per article). If an article cannot be
read, its summary is kept so you still get `k` documents.

## Async

Every tool and the retriever support `ainvoke`; the retriever reads full articles in parallel.

```python
docs = await NestDaddyNewsRetriever(k=5, full_text=True).ainvoke("AI regulation")
```

## Errors

API errors raise `NestDaddyAPIError` with the HTTP status and a hint (a bad key points to the dashboard, a plan
limit to the plans page). Your key is stored as a `SecretStr` and never printed.

## Prefer MCP?

The same tools are available as a remote MCP server for Claude, Cursor, VS Code and n8n:
`https://mcp.nestdaddy.com/mcp`. Setup: [nestdaddy.com/mcp](https://nestdaddy.com/mcp).

## Development

```bash
pip install -e ".[test]"
pytest tests/unit_tests --disable-socket --allow-unix-socket   # offline
NESTDADDY_API_KEY=... pytest -m integration tests/integration_tests
```

MIT licensed.

## Community

Questions, ideas, or something you built? Join **Agent Builders**, our Discord for people building AI agents and MCP servers: daily MCP news, alerts when MCP servers change their tools, and help from our team.

[Join on Discord](https://discord.gg/CXStfSFC3v)
