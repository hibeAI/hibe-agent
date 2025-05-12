"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, Dict, List, Optional, cast

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from react_agent.configuration import Configuration


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


async def get_business_data(key: str = "") -> Dict[str, Any]:
    """Access the company's business data.
    
    This tool provides access to company business metrics including sales figures, 
    customer data, product performance, and regional breakdowns.
    
    Args:
        key: Optional specific data key to retrieve (e.g., "sales", "customers").
            If empty, returns all business data.
    """
    from react_agent.state import State
    
    # This is a placeholder implementation that would normally 
    # access the state data from the actual agent runtime
    business_data = {
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
    
    if key and key in business_data:
        return {key: business_data[key]}
    
    return business_data


# Define which tools are available to which agents
TEAM_LEADER_TOOLS: List[Callable[..., Any]] = [search]
BUSINESS_AGENT_TOOLS: List[Callable[..., Any]] = [get_business_data]

# Default tools for backward compatibility
TOOLS: List[Callable[..., Any]] = [search]
