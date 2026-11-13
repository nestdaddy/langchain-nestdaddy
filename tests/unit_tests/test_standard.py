"""LangChain standard unit tests for the NestDaddy tools (no network)."""

from typing import Type

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_nestdaddy import NestDaddyNewsSearch, NestDaddyReadPage, NestDaddySearch


class TestNestDaddySearchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[NestDaddySearch]:
        return NestDaddySearch

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-key", "max_results": 3}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "model context protocol"}

    @property
    def init_from_env_params(self) -> tuple:
        return ({"NESTDADDY_API_KEY": "env-key"}, {}, {})


class TestNestDaddyNewsSearchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[NestDaddyNewsSearch]:
        return NestDaddyNewsSearch

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "interest rates", "country": "US", "days": 7}


class TestNestDaddyReadPageUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[NestDaddyReadPage]:
        return NestDaddyReadPage

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"url": "https://example.com/article"}
