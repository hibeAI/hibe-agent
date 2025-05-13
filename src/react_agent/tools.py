"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import mysql.connector
import json
import os
from dotenv import load_dotenv
from typing import Any, Callable, Dict, List, Optional, cast, Annotated, Union
import inspect
from pathlib import Path

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool, BaseTool

from react_agent.configuration import Configuration

# Load environment variables from .env
load_dotenv()

# -----------------------------
# Python REPL Tool Definition
# -----------------------------

_python_repl = PythonREPL()


def run_python_code(code: str) -> str:
    """Run arbitrary python code and return the result."""
    try:
        result = _python_repl.run(code)
        return result
    except Exception as e:
        error_msg = f"Error executing Python code: {str(e)}"
        return error_msg


# Define a simple tool that wraps the python repl function
python_repl_tool = Tool(
    name="python_repl",
    func=run_python_code,
    description="Run Python code to interact with the database or perform calculations. The code will be executed in a secure environment. If you want to see output, use print() statements.",
    return_direct=False
)


# def get_db_connection():
#     """Get a MySQL database connection using environment variables."""
#     try:
#         connection = mysql.connector.connect(
#             host=os.getenv('DB_HOST'),
#             user=os.getenv('DB_USER'),
#             password=os.getenv('DB_PASSWORD'),
#             database=os.getenv('DB_NAME'),
#             port=int(os.getenv('DB_PORT', '3306'))
#         )
#         return connection
#     except Exception as e:
#         print(f"\nError creating database connection:")
#         print(f"Type: {type(e).__name__}")
#         print(f"Message: {str(e)}")
#         raise

async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

# Define which tools are available to which agents
TEAM_LEADER_TOOLS: List[Callable[..., Any]] = [search]
JOBS_AGENT_TOOLS: List[Tool] = [python_repl_tool]

# Default tools for backward compatibility
TOOLS: List[Callable[..., Any]] = [search]
