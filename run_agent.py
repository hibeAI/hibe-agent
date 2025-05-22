#!/usr/bin/env python
"""Run the multi-agent system interactively from the command line."""

import asyncio
import os
import time
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage

from src.react_agent.graph import graph, get_graph
from src.react_agent.state import InputState, State

# Load environment variables
load_dotenv()

# Global variable to track the current thread ID
current_thread_id = None

async def run_agent_with_retry(input_state, config, max_retries=3, initial_backoff=2):
    """Run the agent with automatic retries for API overload errors."""
    retries = 0
    backoff = initial_backoff
    
    while retries <= max_retries:
        try:
            # Get graph with checkpointer initialized in async context
            async_graph = await get_graph()
            return await async_graph.ainvoke(input_state, config=config)
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


async def run_agent_with_query(query: str, thread_id: str = None) -> str:
    """Run the agent with a user query and print the response.
    
    Args:
        query (str): The user's input query
        thread_id (str, optional): The conversation thread ID for maintaining context
        
    Returns:
        str: The thread_id used for this conversation
    """
    # Use provided thread_id or generate a new one
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        print(f"Starting new conversation with thread_id: {thread_id}")
        
        # For new threads, create a fresh input with just the current message
        input_state = InputState(
            messages=[HumanMessage(content=query)]
        )
    else:
        print(f"Continuing conversation with thread_id: {thread_id}")
        
        # For existing threads, we need to:
        # 1. Get the async graph implementation with its checkpointer
        async_graph = await get_graph()
        
        # 2. Load the latest state from the checkpointer
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        try:
            # Try to get the last state from the checkpointer
            latest_state = await async_graph.aget_state(config)
            
            # Add the new message to the existing messages
            if latest_state and hasattr(latest_state, "values") and "messages" in latest_state.values:
                # Create input state with all previous messages plus the new one
                input_state = InputState(
                    messages=latest_state.values["messages"] + [HumanMessage(content=query)]
                )
                print(f"Loaded {len(latest_state.values['messages'])} previous messages from history")
            else:
                # Fallback if we can't get previous messages
                input_state = InputState(
                    messages=[HumanMessage(content=query)]
                )
        except Exception as e:
            print(f"Error loading previous state: {e}. Starting with just the current query.")
            input_state = InputState(
                messages=[HumanMessage(content=query)]
            )
    
    # Create configuration with thread_id for memory persistence
    config: Dict = {
        "configurable": {
            "model": os.environ.get("LLM_MODEL", "anthropic/claude-3-7-sonnet-latest"),
            "max_search_results": 5,
            "thread_id": thread_id  # This is the key for memory to work!
        }
    }
    
    # Execute the graph
    print("\nProcessing your query...\n")
    try:
        # Use the retry function to handle overloaded errors
        state = await run_agent_with_retry(input_state, config)
        
        # Extract and display the response
        response_content = None
        
        if hasattr(state, "messages") and state.messages:
            # Get the last AI message
            ai_messages = [msg for msg in state.messages if isinstance(msg, AIMessage)]
            if ai_messages:
                last_ai_message = ai_messages[-1]
                response_content = last_ai_message.content
                print(f"Agent: {response_content}")
            else:
                print("No AI messages found in the state.messages attribute.")
        elif isinstance(state, dict) and "messages" in state:
            # If the state is returned as a dict
            messages = state["messages"]
            if messages and len(messages) > 0:
                last_message = messages[-1]
                response_content = last_message.content
                print(f"Agent: {response_content}")
            else:
                print("No messages found in the state dictionary.")
        else:
            print("No messages found in the response state.")
        
        # Return the thread_id for continuity
        return thread_id
            
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()
        return thread_id


async def interactive_loop() -> None:
    """Run an interactive loop to chat with the agent."""
    global current_thread_id
    
    print("\n===== Multi-Agent System =====")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'new' to start a new conversation thread.\n")
    
    while True:
        # Get user input
        query = input("You: ")
        
        # Check if user wants to exit
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Check if user wants to start a new conversation
        if query.lower() == "new":
            current_thread_id = None
            print("Starting a new conversation thread.")
            continue
        
        # Run the agent with the query, maintaining thread context
        current_thread_id = await run_agent_with_query(query, current_thread_id)


if __name__ == "__main__":
    # Check if a query is provided as a command-line argument
    import sys
    
    if len(sys.argv) > 1:
        # Join all arguments as a single query
        query = " ".join(sys.argv[1:])
        asyncio.run(run_agent_with_query(query))
    else:
        # Run interactive loop
        asyncio.run(interactive_loop()) 