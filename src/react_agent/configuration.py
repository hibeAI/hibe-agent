"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional
import uuid

from langchain_core.runnables import ensure_config
from langgraph.config import get_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    thread_id: Optional[str] = field(
        default=None,
        metadata={
            "description": "Unique identifier for the conversation thread. "
            "If not provided, a random UUID will be generated. "
            "Use the same thread_id to continue a conversation across invocations."
        },
    )

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

    jobs_agent_prompt: str = field(
        default=prompts.JOBS_AGENT_PROMPT,
        metadata={
            "description": "The system prompt for the Jobs Agent. "
            "This agent provides information about specific jobs through database access."
        },
    )

    business_metrics_agent_prompt: str = field(
        default=prompts.BUSINESS_METRICS_AGENT_PROMPT,
        metadata={
            "description": "The system prompt for the Business Metrics Agent. "
            "This agent provides thorough business analysis and recommendations."
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

    max_message_tokens: int = field(
        default=4000,
        metadata={
            "description": "Maximum number of tokens to include in trimmed message history."
        },
    )

    # Database configuration
    mysql_host: str = field(
        default="localhost",
        metadata={
            "description": "MySQL database host",
            "env_var": "DB_HOST"
        },
    )

    mysql_user: str = field(
        default="root",
        metadata={
            "description": "MySQL database user",
            "env_var": "DB_USER"
        },
    )

    mysql_password: str = field(
        default="",
        metadata={
            "description": "MySQL database password",
            "env_var": "DB_PASSWORD"
        },
    )

    mysql_database: str = field(
        default="",
        metadata={
            "description": "MySQL database name",
            "env_var": "MYSQL_DB"
        },
    )

    mysql_port: int = field(
        default=3306,
        metadata={
            "description": "MySQL database port",
            "env_var": "DB_PORT"
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
        
        # Generate a random thread_id if not provided
        if "thread_id" not in configurable:
            configurable["thread_id"] = str(uuid.uuid4())
            
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
