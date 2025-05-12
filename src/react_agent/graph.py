"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import asyncio
from datetime import UTC, datetime
import json
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import BUSINESS_AGENT_TOOLS, TEAM_LEADER_TOOLS, TOOLS, get_business_data
from react_agent.utils import load_chat_model

# Add retry helper function for model calls


async def call_model_with_retry(model, messages, max_retries=3, initial_backoff=2):
    """Call the model with automatic retries for API overload errors."""
    retries = 0
    backoff = initial_backoff
    
    while retries <= max_retries:
        try:
            return await model.ainvoke(messages)
        except Exception as e:
            error_str = str(e)
            # Check if this is an overloaded error
            if "overloaded_error" in error_str and retries < max_retries:
                retries += 1
                wait_time = backoff * (2 ** (retries - 1))  # Exponential backoff
                print(f"\nAPI overloaded in model call. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                # Re-raise if it's not an overloaded error or we've exhausted retries
                raise

# Define the function that calls the team leader agent


async def call_team_leader(state: State) -> Dict:
    """Call the Team Leader agent to evaluate and route queries.

    This agent can search the web and decide whether to handle queries itself
    or route them to specialized agents.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    print("Calling Team Leader agent...")
    configuration = Configuration.from_context()

    # Set current agent in state
    state.current_agent = "team_leader"

    # Initialize the model with tool binding for the team leader
    model = load_chat_model(configuration.model).bind_tools(TEAM_LEADER_TOOLS)

    # Format the team leader system prompt
    system_message = configuration.team_leader_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response with retry
    response = cast(
        AIMessage,
        await call_model_with_retry(
            model, 
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )
    
    print(f"Team Leader response: {response.content[:100]}...")

    # Return the model's response
    return {"messages": [response]}


async def call_business_agent(state: State) -> Dict:
    """Call the Business Agent to answer business-related questions.

    This agent has access to company business data and can answer queries about
    sales, customers, products, and performance metrics.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    print("Calling Business Agent...")
    configuration = Configuration.from_context()

    # Set current agent in state
    state.current_agent = "business_agent"
    
    # Make sure intermediate_responses is empty for this new run
    state.intermediate_responses = []

    # Initialize the model with business data tool access
    model = load_chat_model(configuration.model).bind_tools(BUSINESS_AGENT_TOOLS)

    # Format the business agent system prompt
    system_message = configuration.business_agent_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Create a human message to forward the user's query to the business agent
    last_user_message = next(
        (msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)),
        None,
    )
    if not last_user_message:
        response = AIMessage(content="No user message found to forward to the Business Agent.")
        return {"messages": [response]}

    # First, get the initial response from the business agent
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": last_user_message.content},
    ]
    
    response = cast(
        AIMessage,
        await call_model_with_retry(model, messages),
    )
    
    print(f"Initial Business Agent response: {response}")
    
    # Check if the business agent is trying to call a tool
    if response.tool_calls:
        print(f"Business Agent is calling tools: {response.tool_calls}")
        
        # Process each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            print(f"Executing tool: {tool_name} with args: {tool_args}")
            
            # Execute the appropriate tool
            if tool_name == "get_business_data":
                key = tool_args.get("key", "")
                print(f"Fetching business data with key: {key}")
                tool_result = await get_business_data(key)
                
                # Create a tool message with the result
                tool_message = ToolMessage(
                    content=json.dumps(tool_result, indent=2),
                    tool_call_id=tool_call.get("id"),
                    name=tool_name,
                )
                
                # Add the tool result to the messages
                messages.append(response)
                messages.append(tool_message)
                
                # Get final response after tool use
                final_response = cast(
                    AIMessage,
                    await call_model_with_retry(model, messages),
                )
                
                print(f"Final Business Agent response: {final_response.content[:100]}...")
                
                # Store the complete response for later synthesis
                state.intermediate_responses.append(final_response.content)
                print(f"Added response to intermediate_responses, now has {len(state.intermediate_responses)} items")
                
                # Return both the tool request, tool response, and final agent response
                return {
                    "messages": [response, tool_message, final_response],
                    "intermediate_responses": state.intermediate_responses  # Explicitly return the updated state
                }
    
    # If no tool calls, just return the initial response
    print(f"Business Agent response (no tools): {response.content[:100]}...")
    state.intermediate_responses.append(response.content)
    print(f"Added response to intermediate_responses, now has {len(state.intermediate_responses)} items")
    
    return {
        "messages": [response],
        "intermediate_responses": state.intermediate_responses  # Explicitly return the updated state
    }


async def call_synthesizer(state: State) -> Dict:
    """Call the Synthesizer Agent to create a final, cohesive response.

    This agent takes the outputs from other agents and synthesizes them
    into a user-friendly final answer.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the synthesized response message.
    """
    print("Calling Synthesizer Agent...")
    print(f"Current state intermediate_responses: {state.intermediate_responses}")
    
    configuration = Configuration.from_context()

    # Set current agent in state
    state.current_agent = "synthesizer"

    # Initialize the model (no tools needed for synthesizer)
    model = load_chat_model(configuration.model)

    # Format the synthesizer system prompt
    system_message = configuration.synthesizer_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Create a message containing all the information to synthesize
    # Include the original user query and responses from other agents
    last_user_message = next(
        (msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)),
        None,
    )
    
    if not last_user_message:
        response = AIMessage(content="No user message found for the Synthesizer to process.")
        return {"messages": [response]}
    
    # Get all business agent responses and dump the state for debugging
    business_responses = state.intermediate_responses
    if business_responses:
        print(f"Business responses to synthesize: {len(business_responses)}")
        for i, resp in enumerate(business_responses):
            print(f"Response {i+1} (preview): {resp[:100]}...")
    else:
        print("No business responses to synthesize")
        # Try to extract business agent response from messages
        print("Attempting to extract business data from messages...")
        business_responses = []
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage) and msg.content and len(msg.content) > 0:
                if "sales" in msg.content.lower() and any(year in msg.content for year in ["2023", "2022", "2021"]):
                    print(f"Found potential business data in message: {msg.content[:100]}...")
                    business_responses.append(msg.content)
                    break
    
    # Use either intermediate_responses or extracted message
    synthesis_content = state.intermediate_responses[0] if state.intermediate_responses else (
        business_responses[0] if business_responses else "No information available."
    )

    synthesis_prompt = f"""
Original user question: {last_user_message.content}

Information collected:
{synthesis_content}

Please synthesize this information into a clear, concise response that directly answers the user's question.
"""

    # Get the synthesized response with retry
    response = cast(
        AIMessage,
        await call_model_with_retry(
            model,
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": synthesis_prompt},
            ]
        ),
    )
    
    print(f"Synthesizer response: {response.content[:100]}...")

    # Reset intermediate responses for next conversation
    state.intermediate_responses = []

    # Return the synthesized response
    return {"messages": [response]}


# Legacy function for backward compatibility
async def call_model(state: State) -> Dict:
    """Call the LLM powering our "agent" (legacy function).

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response with retry
    response = cast(
        AIMessage,
        await call_model_with_retry(
            model,
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define the routing function to determine next step based on message content
def route_team_leader_output(state: State) -> Literal["__end__", "business_agent", "team_leader_tools"]:
    """Determine where to route based on the Team Leader's decision.

    This function analyzes the last message to determine if:
    1. The query should be routed to the Business Agent
    2. The Team Leader needs to use tools
    3. The process should end with a direct answer
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        str: The name of the next node to call.
    """
    if not state.messages:
        print("No messages in state!")
        return "__end__"
        
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        print(f"Expected AIMessage in output edges, but got {type(last_message).__name__}")
        return "__end__"
    
    # If there are tool calls, use tools
    if last_message.tool_calls:
        print("Routing to tools...")
        return "team_leader_tools"
    
    # Check if this is a business query by looking for keywords
    user_message = next(
        (msg.content.lower() for msg in reversed(state.messages) if isinstance(msg, HumanMessage)),
        ""
    )
    
    business_keywords = ["sales", "revenue", "customer", "product", "business", 
                        "profit", "performance", "quarter", "fiscal", "market",
                        "growth", "year", "income", "company", "financial"]
    
    if any(keyword in user_message for keyword in business_keywords):
        print("Routing to business agent...")
        return "business_agent"
    
    # If not a business query and no tool calls, end the process
    print("Routing to end (direct response)...")
    return "__end__"


# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add nodes for each agent and tools
builder.add_node("team_leader", call_team_leader)
builder.add_node("business_agent", call_business_agent)
builder.add_node("synthesizer", call_synthesizer)
builder.add_node("team_leader_tools", ToolNode(TEAM_LEADER_TOOLS))

# Set the entrypoint as the team leader
builder.add_edge("__start__", "team_leader")

# Add conditional edges for team leader
builder.add_conditional_edges(
    "team_leader",
    route_team_leader_output,
)

# From business agent, always go to synthesizer
builder.add_edge("business_agent", "synthesizer")

# After using tools, go back to team leader
builder.add_edge("team_leader_tools", "team_leader")

# Synthesizer output ends the process
builder.add_edge("synthesizer", "__end__")

# Compile the builder into an executable graph
graph = builder.compile(name="Multi-Agent Team")
