import asyncio
from langgraph_sdk import get_client

async def test_prod_agent():
    client = get_client(url="https://hibe-agent-53b54d9dbefc5e06a0217cb9bfc4511f.us.langgraph.app", api_key="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")

    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

if __name__ == "__main__":
    asyncio.run(test_prod_agent()) 