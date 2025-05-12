"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    current_agent: Literal["team_leader", "business_agent", "synthesizer"] = field(
        default="team_leader"
    )
    """Tracks which agent is currently active in the workflow."""

    business_data: Dict[str, any] = field(
        default_factory=lambda: {
            "sales": {
                "2023": 1250000,
                "2022": 980000,
                "2021": 750000,
            },
            "customers": {
                "total": 350,
                "active": 280,
                "new_last_year": 75,
            },
            "products": [
                {"name": "Product A", "sales_2023": 450000},
                {"name": "Product B", "sales_2023": 325000},
                {"name": "Product C", "sales_2023": 475000},
            ],
            "regions": {
                "North": {"sales_2023": 400000},
                "South": {"sales_2023": 350000},
                "East": {"sales_2023": 200000},
                "West": {"sales_2023": 300000},
            }
        }
    )
    """Dummy business data for the business agent to access."""

    intermediate_responses: List[str] = field(default_factory=list)
    """Stores responses from agents before final synthesis."""

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
