import asyncio
from langgraph_sdk import get_client

async def create_thread(client):
    # Create a new thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    return thread_id

async def test_prod_agent():
    # Connect to the LangGraph app
    client = get_client(url="https://hibe-agent-53b54d9dbefc5e06a0217cb9bfc4511f.us.langgraph.app", api_key="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")
    
    # Create a thread for this conversation
    thread_id = await create_thread(client)
    print(f"Created new thread with ID: {thread_id}")
    
    # Use the thread ID in your run
    async for chunk in client.runs.stream(
        thread_id,  # Use the thread ID we created
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "My sales in 2023 where $1m and in 2024 they were $2m. What is the growth rate?"  # Simple string content
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
        
    # You can continue the conversation in the same thread by starting another run
    # This allows for maintaining context between interactions
    
    # # Example of continuing the conversation in the same thread
    # async for chunk in client.runs.stream(
    #     thread_id,  # Same thread ID to continue conversation
    #     "agent",
    #     input={
    #         "messages": [{
    #             "role": "human",
    #             "content": "Can you give me more details about its features?"
    #         }],
    #     },
    #     stream_mode="updates",
    # ):
    #     print(f"Continuation - New event of type: {chunk.event}...")
    #     print(chunk.data)
    #     print("\n\n")

if __name__ == "__main__":
    asyncio.run(test_prod_agent()) 