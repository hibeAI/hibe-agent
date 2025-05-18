"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import asyncio
from datetime import UTC, datetime
import json
from typing import Dict, List, Literal, cast
import sqlite3
import os
import aiosqlite
from openai import OpenAI

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, AnyMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import (
    JOBS_AGENT_TOOLS,
    TEAM_LEADER_TOOLS,
    TOOLS,
)
from react_agent.utils import load_chat_model

# Create a synchronous checkpointer for reference and non-async operations
# Using SQLite for persistent storage across restarts
sqlite_conn = sqlite3.connect("agent_state.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)

# Initialize the OpenAI client
client = OpenAI()

# Define EventHandler for OpenAI assistant streaming
class EventHandler:
    def __init__(self):
        """Initialize the event handler with an empty responses list."""
        self.responses = []
        
    def on_text_created(self, text):
        """Handler for text creation events."""
        print(text.value, end="", flush=True)
        self.responses.append(text.value)
        
    def on_text_delta(self, delta, snapshot):
        """Handler for text delta events."""
        print(delta.value, end="", flush=True)
        
    def on_tool_call_created(self, tool_call):
        """Handler for tool call creation events."""
        print(f"\n[Tool Call: {tool_call.type}]", flush=True)
        
    def on_tool_call_delta(self, delta, snapshot):
        """Handler for tool call delta events."""
        if delta.type == "code_interpreter":
            if hasattr(delta, "input") and delta.input:
                print(delta.input, end="", flush=True)
            if hasattr(delta, "outputs") and delta.outputs:
                for output in delta.outputs:
                    if output.type == "logs":
                        print(f"\n[Code Output]\n{output.logs}\n", flush=True)
    
    # Required methods for OpenAI client compatibility
    def on_event(self, event):
        """Generic event handler that dispatches to specific handlers."""
        event_type = event.get("type")
        if event_type == "thread.message.created":
            self._handle_message_created(event.get("data", {}))
        elif event_type == "thread.message.delta":
            self._handle_message_delta(event.get("data", {}))
        elif event_type == "thread.run.created":
            pass  # Handle if needed
        elif event_type == "thread.run.queued":
            pass  # Handle if needed
        elif event_type == "thread.run.in_progress":
            pass  # Handle if needed
        elif event_type == "thread.run.completed":
            pass  # Handle if needed
        elif event_type == "thread.run.failed":
            print(f"\nRun failed: {event.get('data', {}).get('error', 'Unknown error')}")
        elif event_type == "thread.run.cancelled":
            print("\nRun cancelled")
            
    def _handle_message_created(self, data):
        """Handle message created events."""
        message = data.get("object")
        if message and message.get("role") == "assistant":
            for content in message.get("content", []):
                if content.get("type") == "text":
                    text_value = content.get("text", {}).get("value", "")
                    if text_value:
                        print(text_value, end="", flush=True)
                        self.responses.append(text_value)
                        
    def _handle_message_delta(self, data):
        """Handle message delta events."""
        delta = data.get("delta", {})
        for content in delta.get("content", []):
            if content.get("type") == "text":
                text_value = content.get("text", {}).get("value", "")
                if text_value:
                    print(text_value, end="", flush=True)
    
    def get_responses(self):
        """Return all collected responses."""
        return self.responses

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
    
    # --- Ensure we don't leave dangling tool_use messages without their corresponding tool_result ---
    def _remove_unpaired_tool_messages(msgs):
        """Remove AIMessage entries that contain `tool_calls` if their corresponding
        ToolMessage result is not present immediately after. This avoids Anthropic
        API 400 errors like:
        `tool_use ids were found without tool_result blocks immediately after`.

        If a ToolMessage appears without the preceding AIMessage (because it was
        trimmed), it is also dropped.
        """
        cleaned: List[AnyMessage] = []
        idx = 0
        while idx < len(msgs):
            msg = msgs[idx]
            # Handle AIMessage that invoked a tool
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                if idx + 1 < len(msgs) and isinstance(msgs[idx + 1], ToolMessage):
                    # Keep both the tool_use and its tool_result
                    cleaned.append(msg)
                    cleaned.append(msgs[idx + 1])
                    idx += 2
                    continue
                # Otherwise, drop the unpaired tool_use (and any stray next message)
                idx += 1
                continue
            # Drop stray ToolMessage without its preceding AIMessage
            if isinstance(msg, ToolMessage):
                idx += 1
                continue
            # Keep all other messages
            cleaned.append(msg)
            idx += 1
        return cleaned

    trimmed_messages = _remove_unpaired_tool_messages(trimmed_messages)
    
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

    # Get all user messages to include context
    user_messages = []
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            user_messages.append(msg.content)
            
    if not user_messages:
        response = AIMessage(content="No user messages found to forward to the Jobs Agent.")
        return {"messages": [response]}
    
    # Create a context-rich query with history
    context_message = ""
    if len(user_messages) > 1:
        # Include previous messages as context
        context_message = "User conversation history:\n"
        for i, msg in enumerate(user_messages[:-1]):
            context_message += f"Message {i+1}: {msg}\n"
        context_message += f"\nCurrent user query: {user_messages[-1]}\n\n"
        context_message += "Please analyze the full conversation history when answering the current query."
    else:
        # Just one message
        context_message = user_messages[0]
    
    # Apply pre-model hook to manage message history
    hook_result = await pre_model_hook(state)
    input_messages = hook_result.get("llm_input_messages", state.messages)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": context_message},
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
                    
                    # Add result context to user message - without asking to "continue answering"
                    context_msg = f"""
Based on your previous request, I've executed your database query. Here are the results:

{tool_result}

Please analyze these results and provide the answer to the user's original question.
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
    thorough business insights and recommendations using an OpenAI Assistant.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    print("Calling Business Metrics Agent (OpenAI Assistant)...")
    
    # Set current agent in state
    state.current_agent = "business_metrics_agent"
    
    # Make sure intermediate_responses is empty for this new run
    state.intermediate_responses = []

    # Get the Assistant ID from environment variables
    ASSISTANT_ID = os.getenv('assistant_id')
    if not ASSISTANT_ID:
        response = AIMessage(content="Error: OpenAI Assistant ID not found in environment variables.")
        return {"messages": [response], "intermediate_responses": [response.content]}

    # Get all user messages to include context
    user_messages = []
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            user_messages.append(msg.content)
            
    if not user_messages:
        response = AIMessage(content="No user messages found to forward to the Business Metrics Agent.")
        return {"messages": [response], "intermediate_responses": [response.content]}
    
    # Create a context-rich query with history
    context_message = ""
    if len(user_messages) > 1:
        # Include previous messages as context
        context_message = "User conversation history:\n"
        for i, msg in enumerate(user_messages[:-1]):
            context_message += f"Message {i+1}: {msg}\n"
        context_message += f"\nCurrent user query: {user_messages[-1]}\n\n"
        context_message += "Please analyze the full conversation history when answering the current query."
    else:
        # Just one message
        context_message = user_messages[0]
    
    try:
        # Create a thread
        thread = client.beta.threads.create()
        
        # Add a message to the thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=context_message
        )
        
        # Run the assistant and wait for completion - no streaming
        print("\nProcessing your request with Business Metrics Agent...\n")
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
        )
        
        # Poll for completion
        while run.status in ["queued", "in_progress"]:
            await asyncio.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
        # Retrieve messages after completion
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        
        # Get the assistant's responses
        assistant_responses = []
        for msg in messages.data:
            if msg.role == "assistant":
                for content_part in msg.content:
                    if content_part.type == "text":
                        assistant_responses.append(content_part.text.value)
        
        if not assistant_responses:
            response = AIMessage(content="No response received from the Business Metrics Agent.")
            return {"messages": [response], "intermediate_responses": [response.content]}
            
        # Create a single response from all assistant messages
        response_content = "\n\n".join(assistant_responses)
        print(f"\nBusiness Metrics Agent response: {response_content[:100]}...\n")
        
        response = AIMessage(content=response_content)
        
        # Store response for synthesis
        state.intermediate_responses = [response_content]
        
        # Return the response
        return {
            "messages": [response],
            "intermediate_responses": [response_content]
        }
        
    except Exception as e:
        error_msg = f"Error in Business Metrics Agent (OpenAI Assistant): {str(e)}"
        print(error_msg)
        response = AIMessage(content=f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or contact support if the issue persists.")
        state.intermediate_responses = [response.content]
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

# Compile the regular graph with the synchronous checkpointer for non-async use
graph = builder.compile(name="Multi-Agent Team", checkpointer=checkpointer)

# Create a helper function to get the graph with async checkpointer
async def get_graph():
    """Get the graph with an async-compatible checkpointer.
    
    This maintains the same database file as the synchronous checkpointer,
    ensuring conversation history is preserved across both versions.
    
    Returns:
        The compiled graph with async checkpointer.
    """
    # Create the async SQLite connection to the SAME database file
    # This ensures conversation history is shared between sync and async checkpointers
    aiosqlite_conn = await aiosqlite.connect("agent_state.sqlite")
    
    # Create the async checkpointer with the same database
    async_checkpointer = AsyncSqliteSaver(aiosqlite_conn)
    
    # Return the graph with the async checkpointer
    return builder.compile(name="Multi-Agent Team", checkpointer=async_checkpointer)
