"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import ensure_config
from langgraph.config import get_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    team_leader_prompt: str = field(
        default=prompts.TEAM_LEADER_PROMPT,
        metadata={
            "description": "The system prompt for the Team Leader agent. "
            "This agent evaluates queries and routes them to specialized agents."
        },
    )

    business_agent_prompt: str = field(
        default=prompts.BUSINESS_AGENT_PROMPT,
        metadata={
            "description": "The system prompt for the Business Agent. "
            "This agent provides information about business metrics and KPIs."
        },
    )

    synthesizer_prompt: str = field(
        default=prompts.SYNTHESIZER_PROMPT,
        metadata={
            "description": "The system prompt for the Synthesizer Agent. "
            "This agent creates a cohesive final response from multiple sources."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
