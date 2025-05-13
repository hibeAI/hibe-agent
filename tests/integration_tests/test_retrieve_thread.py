import asyncio
from langgraph_sdk import get_client
import sys

async def retrieve_and_use_thread():
    # Connect to the LangGraph app
    client = get_client(url="https://hibe-agent-53b54d9dbefc5e06a0217cb9bfc4511f.us.langgraph.app", api_key="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")
    
    # Check if thread_id is provided as command line argument
    if len(sys.argv) < 2:
        print("Error: Thread ID is required")
        print("Usage: python test_retrieve_thread.py <thread_id>")
        return
    
    # Retrieve the existing thread
    thread_id = sys.argv[1]
    try:
        thread = await client.threads.get(thread_id=thread_id)
        print(f"Successfully retrieved thread: {thread_id}")
        
        # Optionally, you could list previous messages in the thread here
        # messages = await client.threads.messages.list(thread_id=thread_id)
        # print(f"Thread contains {len(messages)} messages")
        
    except Exception as e:
        print(f"Error retrieving thread: {e}")
        return
    
    # Continue the conversation in the retrieved thread
    print(f"Continuing conversation in thread: {thread_id}")
    async for chunk in client.runs.stream(
        thread_id,
        "agent",
        input={
            "messages": [{
                "role": "human",
                "content": "Let's continue the conversation. What else can you tell me?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    
    print(f"Conversation in thread {thread_id} completed")

if __name__ == "__main__":
    asyncio.run(retrieve_and_use_thread()) 