#!/usr/bin/env python
"""Run the multi-agent system interactively from the command line."""

import asyncio
import os
import time
from typing import Dict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.react_agent.graph import graph
from src.react_agent.state import InputState

# Load environment variables
load_dotenv()


async def run_agent_with_retry(input_state, config, max_retries=3, initial_backoff=2):
    """Run the agent with automatic retries for API overload errors."""
    retries = 0
    backoff = initial_backoff
    
    while retries <= max_retries:
        try:
            return await graph.ainvoke(input_state, config=config)
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


async def run_agent_with_query(query: str) -> None:
    """Run the agent with a user query and print the response."""
    # Create input state with the query
    input_state = InputState(
        messages=[HumanMessage(content=query)]
    )
    
    # Create configuration with default values
    config: Dict = {
        "configurable": {
            "model": os.environ.get("LLM_MODEL", "anthropic/claude-3-5-sonnet-20240620"),
            "max_search_results": 5,
        }
    }
    
    # Execute the graph
    print("\nProcessing your query...\n")
    try:
        # Use the retry function to handle overloaded errors
        state = await run_agent_with_retry(input_state, config)
        
        # Add debugging to inspect the state
        print(f"State type: {type(state)}")
        print(f"State attributes: {dir(state)}")
        
        # Try different ways to access messages
        if hasattr(state, "messages") and state.messages:
            # Get the last AI message
            ai_messages = [msg for msg in state.messages if isinstance(msg, AIMessage)]
            if ai_messages:
                last_ai_message = ai_messages[-1]
                print(f"Agent: {last_ai_message.content}")
            else:
                print("No AI messages found in the state.messages attribute.")
        elif isinstance(state, dict) and "messages" in state:
            # If the state is returned as a dict
            messages = state["messages"]
            if messages and len(messages) > 0:
                print(f"Agent: {messages[-1].content}")
            else:
                print("No messages found in the state dictionary.")
        elif isinstance(state, dict):
            # Debug what keys are available
            print(f"State keys: {state.keys()}")
            # Try to find messages in any form
            for key, value in state.items():
                if "message" in key.lower():
                    print(f"Found potential messages in key: {key}")
                    print(f"Value: {value}")
        else:
            print("No messages found in the response state.")
            
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()


async def interactive_loop() -> None:
    """Run an interactive loop to chat with the agent."""
    print("\n===== Multi-Agent System =====")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        # Get user input
        query = input("You: ")
        
        # Check if user wants to exit
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Run the agent with the query
        await run_agent_with_query(query)


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