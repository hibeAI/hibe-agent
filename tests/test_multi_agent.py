"""Test the multi-agent system locally."""

import asyncio
import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from react_agent.configuration import Configuration
from react_agent.graph import graph
from react_agent.state import InputState

# Load environment variables
load_dotenv()


async def run_test(query: str) -> None:
    """Run a test with the given query."""
    print(f"\n\n==== Testing with query: '{query}' ====\n")
    
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
    result = await graph.ainvoke(input_state, config=config)
    
    # Print the result
    print("\n==== RESULTS ====\n")
    
    # Access the messages in the result
    messages = result.messages
    
    # Print all messages
    for i, message in enumerate(messages):
        role = "User" if message.type == "human" else "Agent"
        print(f"{role}: {message.content}")
        
        # If there are tool calls, print them
        if hasattr(message, "tool_calls") and message.tool_calls:
            print("\nTool Calls:")
            for tool_call in message.tool_calls:
                print(f"  Tool: {tool_call['name']}")
                print(f"  Args: {tool_call['args']}")
        
        if i < len(messages) - 1:
            print("-" * 80)
    
    print("\n==== TEST COMPLETE ====\n")


async def main() -> None:
    """Run multiple test cases."""
    # Test cases
    test_queries = [
        "What were our sales in 2023?",  # Business query - should route to business agent
        "Who won the World Cup in 2022?",  # General query - should use web search
        "How many active customers do we have?",  # Business query about customers
    ]
    
    for query in test_queries:
        await run_test(query)


if __name__ == "__main__":
    asyncio.run(main()) 