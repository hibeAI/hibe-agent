"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import asyncio
from datetime import UTC, datetime
import json
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import (
    JOBS_AGENT_TOOLS,
    TEAM_LEADER_TOOLS,
    TOOLS,
)
from react_agent.utils import load_chat_model

# Create a checkpointer for memory persistence
# This needs to be accessible from other modules
checkpointer = InMemorySaver()

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
                print(f"\nAPI overloaded. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                # Re-raise if it's not an overloaded error or we've exhausted retries
                raise

async def pre_model_hook(state: State) -> Dict:
    """Manage message history by trimming when it gets too long.
    
    This prevents the context window from being exceeded during long conversations.
    
    Args:
        state (State): The current conversation state.
        
    Returns:
        Dict: Updated state with trimmed messages for LLM input.
    """
    configuration = Configuration.from_context()
    
    # Check if we already have a summary in context
    if "message_summary" not in state.context:
        state.context["message_summary"] = ""
    
    # Trim messages to manage context length
    trimmed_messages = trim_messages(
        state.messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=configuration.max_message_tokens,  # Use configured value
        start_on="human",
        end_on=("human", "tool")
    )
    
    # Return the trimmed messages for LLM input
    return {"llm_input_messages": trimmed_messages}


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
    
    # Apply pre-model hook to manage message history
    hook_result = await pre_model_hook(state)
    input_messages = hook_result.get("llm_input_messages", state.messages)

    # Get the model's response with retry
    response = cast(
        AIMessage,
        await call_model_with_retry(
            model, 
            [{"role": "system", "content": system_message}, *input_messages]
        ),
    )
    
    print(f"Team Leader response: {response.content[:100]}...")

    # Return the model's response
    return {"messages": [response]}


async def call_jobs_agent(state: State) -> Dict:
    """Call the Jobs Agent to answer job-related questions.

    This agent has access to job data and can answer queries about
    specific jobs in the database.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    print("Calling Jobs Agent...")
    configuration = Configuration.from_context()

    # Set current agent in state
    state.current_agent = "jobs_agent"
    
    # Make sure intermediate_responses is empty for this new run
    state.intermediate_responses = []

    # Initialize the model with jobs data tool access
    model = load_chat_model(configuration.model).bind_tools(JOBS_AGENT_TOOLS)

    # Format the jobs agent system prompt
    system_message = configuration.jobs_agent_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Create a human message to forward the user's query to the jobs agent
    last_user_message = next(
        (msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)),
        None,
    )
    if not last_user_message:
        response = AIMessage(content="No user message found to forward to the Jobs Agent.")
        return {"messages": [response]}
    
    # Apply pre-model hook to manage message history
    hook_result = await pre_model_hook(state)
    input_messages = hook_result.get("llm_input_messages", state.messages)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": last_user_message.content},
    ]
    
    all_responses = []
    
    try:
        # Ensure we're using a format compatible with Anthropic API
        clean_initial_messages = []
        for msg in messages:
            if msg["role"] == "system":
                clean_initial_messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                clean_initial_messages.append({"role": "user", "content": msg["content"]})
        
        # Get initial response from the jobs agent with clean message format
        response = cast(
            AIMessage,
            await call_model_with_retry(model, clean_initial_messages),
        )
        
        # Process tool calls until we get a final response without tool calls
        # Make sure tool_calls is actually present and is a list
        while hasattr(response, 'tool_calls') and response.tool_calls and isinstance(response.tool_calls, list):
            # Process each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                # Find the tool from JOBS_AGENT_TOOLS
                tool = None
                for t in JOBS_AGENT_TOOLS:
                    if hasattr(t, "name") and t.name == tool_name:
                        tool = t
                        break
                        
                if tool is None:
                    tool_result = f"Error: Tool '{tool_name}' not found in JOBS_AGENT_TOOLS"
                else:
                    try:
                        # Handle different argument formats
                        if tool_name == "python_repl":
                            # Extract the code from various possible argument formats
                            if "code" in tool_args:
                                code = tool_args["code"]
                            elif "__arg1" in tool_args:
                                code = tool_args["__arg1"]
                            else:
                                # Take the first argument regardless of name
                                code = next(iter(tool_args.values()), "")
                                
                            if callable(tool.func):
                                if asyncio.iscoroutinefunction(tool.func):
                                    tool_result = await tool.func(code)
                                else:
                                    tool_result = tool.func(code)
                            else:
                                tool_result = f"Error: Tool function is not callable"
                        else:
                            # For other tools, pass all arguments
                            if callable(tool.func):
                                if asyncio.iscoroutinefunction(tool.func):
                                    tool_result = await tool.func(**tool_args)
                                else:
                                    tool_result = tool.func(**tool_args)
                            else:
                                tool_result = f"Error: Tool function is not callable"
                    except Exception as e:
                        tool_result = f"Error executing tool: {str(e)}"
              
                # Create a tool message with the result
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call.get("id"),
                    name=tool_name,
                )
                
                # Add the response and tool result to messages - without extra fields that might cause API errors
                assistant_content = response.content
                
                # Only keep the essential fields for API compatibility
                # For Anthropic API, we need to match exactly what it expects for tool calls
                if isinstance(tool_call, dict) and "id" in tool_call:
                    tool_call_id = tool_call["id"]
                else:
                    # Generate a simple ID if none exists
                    tool_call_id = f"call_{len(messages)}"
                    
                messages.extend([
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "tool", 
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": str(tool_result)
                    },
                ])
                
                # Get next response after tool use
                try:
                    # Reset with only system and most recent user message for Claude
                    # This avoids tool history format issues while maintaining context
                    cleaned_messages = [
                        {"role": "system", "content": system_message}
                    ]
                    
                    # Find the last user message
                    last_user_msg = None
                    for msg in messages:
                        if msg["role"] == "user":
                            last_user_msg = msg["content"]
                    
                    # Add result context to user message
                    context_msg = f"""
Based on the database query, here's what I've found:
{tool_result}

Please continue answering my question:
{last_user_msg}
"""
                    cleaned_messages.append({"role": "user", "content": context_msg})
                    
                    response = cast(
                        AIMessage,
                        await call_model_with_retry(model, cleaned_messages),
                    )
                except Exception as e:
                    print(f"Error in tool response handling: {str(e)}")
                    # Create a simple response if there's an error
                    response = AIMessage(content=f"I processed your request but encountered an issue: {str(e)}. Here's what I found so far: {tool_result}")
                
                # Store responses for synthesis
                if response.content:
                    all_responses.append(response.content)
        
        # Add the final response
        all_responses.append(response.content)
        state.intermediate_responses = all_responses
        
    except Exception as e:
        error_msg = f"Error in Jobs Agent tool execution: {str(e)}"
        print(error_msg)
        response = AIMessage(content=f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or contact support if the issue persists.")
        state.intermediate_responses = [response.content]
    
    # Return all messages including tool interactions
    return {
        "messages": [response],
        "intermediate_responses": state.intermediate_responses
    }


async def call_business_metrics_agent(state: State) -> Dict:
    """Call the Business Metrics Agent to provide business analysis.

    This agent specializes in business metrics analysis and provides
    thorough business insights and recommendations.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    print("Calling Business Metrics Agent...")
    configuration = Configuration.from_context()

    # Set current agent in state
    state.current_agent = "business_metrics_agent"
    
    # Make sure intermediate_responses is empty for this new run
    state.intermediate_responses = []

    # Initialize the model (no tools needed for business metrics agent)
    model = load_chat_model(configuration.model)

    # Format the business metrics agent system prompt
    system_message = configuration.business_metrics_agent_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Create a human message to forward the user's query to the business metrics agent
    last_user_message = next(
        (msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)),
        None,
    )
    if not last_user_message:
        response = AIMessage(content="No user message found to forward to the Business Metrics Agent.")
        return {"messages": [response]}
    
    # Create a prompt with just the user question
    prompt = f"""
User question: {last_user_message.content}

Please analyze this question and provide thorough insights and recommendations.
"""
    
    # Apply pre-model hook to manage message history
    hook_result = await pre_model_hook(state)
    
    # Get the business metrics agent's response with retry
    response = cast(
        AIMessage,
        await call_model_with_retry(
            model,
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
        ),
    )
    
    # Store response for synthesis
    state.intermediate_responses = [response.content]
    
    # Return the response
    return {
        "messages": [response],
        "intermediate_responses": [response.content]
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
    
    # Get agent responses
    agent_responses = state.intermediate_responses
    if not agent_responses:
        # Try to extract agent response from messages
        agent_responses = []
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage) and msg.content and len(msg.content) > 0:
                if state.current_agent in ["jobs_agent", "business_metrics_agent"]:
                    agent_responses.append(msg.content)
                    break
    
    # Convert any list items to strings
    processed_responses = []
    for response in agent_responses:
        if isinstance(response, list):
            # Handle lists of items
            response_parts = []
            for item in response:
                if item is None:
                    continue
                elif isinstance(item, dict) and "text" in item:
                    response_parts.append(str(item["text"]))
                else:
                    response_parts.append(str(item))
            processed_responses.append(" ".join(response_parts))
        else:
            processed_responses.append(str(response))
    
    # Use intermediate_responses or extracted message
    synthesis_content = '\n\n'.join(processed_responses) if processed_responses else "No information available."

    synthesis_prompt = f"""
Original user question: {last_user_message.content}

Information collected:
{synthesis_content}

Please synthesize this information into a clear, concise response that directly answers the user's question.
"""

    # Apply pre-model hook to manage message history
    hook_result = await pre_model_hook(state)
    
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
def route_team_leader_output(state: State) -> Literal["__end__", "jobs_agent", "business_metrics_agent", "team_leader_tools"]:
    """Determine where to route based on the Team Leader's decision.

    This function analyzes the last message to determine if:
    1. The query should be routed to the Jobs Agent
    2. The query should be routed to the Business Metrics Agent
    3. The Team Leader needs to use tools
    4. The process should end with a direct answer
    
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

    # Get content and handle potential list format
    content = last_message.content
    
    # Handle case where content is a list (convert to string)
    if isinstance(content, list):
        # Handle both simple lists and lists of objects
        content_parts = []
        for item in content:
            if item is None:
                continue
            elif isinstance(item, dict) and "text" in item:
                content_parts.append(str(item["text"]))
            else:
                content_parts.append(str(item))
        content = " ".join(content_parts)
        print(f"Converted list content to string: {content[:100]}...")
    elif content is None:
        # Handle case where content is None
        content = ""
        print("Content was None, using empty string")
    else:
        # Convert to string if it's any other type
        content = str(content)
    
    # Convert to lowercase for consistent checking
    response_lower = content.lower()
    
    # Check if the Team Leader's response indicates routing to Jobs Agent
    if "jobs agent" in response_lower or "route to jobs" in response_lower:
        print("Team Leader decided to route to Jobs Agent")
        return "jobs_agent"
    
    # Check if the Team Leader's response indicates routing to Business Metrics Agent
    if "business metrics agent" in response_lower or "route to business metrics" in response_lower:
        print("Team Leader decided to route to Business Metrics Agent")
        return "business_metrics_agent"
    
    # If there are tool calls, use tools
    if last_message.tool_calls:
        print("Team Leader wants to use tools")
        return "team_leader_tools"
    
    # If no specific routing or tools needed, end the process
    print("Team Leader provided direct response")
    return "__end__"


# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add nodes for each agent and tools
builder.add_node("team_leader", call_team_leader)
builder.add_node("jobs_agent", call_jobs_agent)
builder.add_node("business_metrics_agent", call_business_metrics_agent)
builder.add_node("synthesizer", call_synthesizer)
builder.add_node("team_leader_tools", ToolNode(TEAM_LEADER_TOOLS))

# Set the entrypoint as the team leader
builder.add_edge("__start__", "team_leader")

# Add conditional edges for team leader
builder.add_conditional_edges(
    "team_leader",
    route_team_leader_output,
)

# From jobs agent and business metrics agent, go to synthesizer
builder.add_edge("jobs_agent", "synthesizer")
builder.add_edge("business_metrics_agent", "synthesizer")

# After using tools, go back to team leader
builder.add_edge("team_leader_tools", "team_leader")

# Synthesizer output ends the process
builder.add_edge("synthesizer", "__end__")

# Compile the builder into an executable graph with checkpointer
graph = builder.compile(name="Multi-Agent Team", checkpointer=checkpointer)
