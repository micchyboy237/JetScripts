import asyncio
from jet.transformers.formatters import format_json
from __future__ import annotations
from agents import (
Agent,
ItemHelpers,
MessageOutputItem,
RunContextWrapper,
Runner,
ToolCallItem,
ToolCallOutputItem,
TResponseInputItem,
function_tool,
)
from jet.logger import CustomLogger
from mem0 import AsyncMemoryClient
from pydantic import BaseModel
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Mem0 as an Agentic Tool
---


Integrate Mem0's memory capabilities with MLX's Agents SDK to create AI agents with persistent memory.
You can create agents that remember past conversations and use that context to provide better responses.

## Installation

First, install the required packages:
"""
logger.info("## Installation")

pip install mem0ai pydantic openai-agents

"""
You'll also need a custom agents framework for this implementation.

## Setting Up Environment Variables

Store your Mem0 API key as an environment variable:
"""
logger.info("## Setting Up Environment Variables")

export MEM0_API_KEY="your_mem0_api_key"

"""
Or in your Python script:
"""
logger.info("Or in your Python script:")

os.environ["MEM0_API_KEY"] = "your_mem0_api_key"

"""
## Code Structure

The integration consists of three main components:

1. **Context Manager**: Defines user context for memory operations
2. **Memory Tools**: Functions to add, search, and retrieve memories
3. **Memory Agent**: An agent configured to use these memory tools

## Step-by-Step Implementation

### 1. Import Dependencies
"""
logger.info("## Code Structure")

try:
except ImportError:
    raise ImportError("mem0 is not installed. Please install it using 'pip install mem0ai'.")

"""
### 2. Define Memory Context
"""
logger.info("### 2. Define Memory Context")

class Mem0Context(BaseModel):
    user_id: str | None = None

"""
### 3. Initialize the Mem0 Client
"""
logger.info("### 3. Initialize the Mem0 Client")

client = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY"))

"""
### 4. Create Memory Tools

#### Add to Memory
"""
logger.info("### 4. Create Memory Tools")

@function_tool
async def add_to_memory(
    context: RunContextWrapper[Mem0Context],
    content: str,
) -> str:
    """
    Add a message to Mem0
    Args:
        content: The content to store in memory.
    """
    messages = [{"role": "user", "content": content}]
    user_id = context.context.user_id or "default_user"
    async def run_async_code_d1de8d7d():
        await client.add(messages, user_id=user_id)
        return 
     = asyncio.run(run_async_code_d1de8d7d())
    logger.success(format_json())
    return f"Stored message: {content}"

"""
#### Search Memory
"""
logger.info("#### Search Memory")

@function_tool
async def search_memory(
    context: RunContextWrapper[Mem0Context],
    query: str,
) -> str:
    """
    Search for memories in Mem0
    Args:
        query: The search query.
    """
    user_id = context.context.user_id or "default_user"
    async def run_async_code_ccc55a3d():
        async def run_async_code_5098027c():
            memories = await client.search(query, user_id=user_id, output_format="v1.1")
            return memories
        memories = asyncio.run(run_async_code_5098027c())
        logger.success(format_json(memories))
        return memories
    memories = asyncio.run(run_async_code_ccc55a3d())
    logger.success(format_json(memories))
    results = '\n'.join([result["memory"] for result in memories["results"]])
    return str(results)

"""
#### Get All Memories
"""
logger.info("#### Get All Memories")

@function_tool
async def get_all_memory(
    context: RunContextWrapper[Mem0Context],
) -> str:
    """Retrieve all memories from Mem0"""
    user_id = context.context.user_id or "default_user"
    async def run_async_code_ea179b5a():
        async def run_async_code_76dfc99f():
            memories = await client.get_all(user_id=user_id, output_format="v1.1")
            return memories
        memories = asyncio.run(run_async_code_76dfc99f())
        logger.success(format_json(memories))
        return memories
    memories = asyncio.run(run_async_code_ea179b5a())
    logger.success(format_json(memories))
    results = '\n'.join([result["memory"] for result in memories["results"]])
    return str(results)

"""
### 5. Configure the Memory Agent
"""
logger.info("### 5. Configure the Memory Agent")

memory_agent = Agent[Mem0Context](
    name="Memory Assistant",
    instructions="""You are a helpful assistant with memory capabilities. You can:
    1. Store new information using add_to_memory
    2. Search existing information using search_memory
    3. Retrieve all stored information using get_all_memory
    When users ask questions:
    - If they want to store information, use add_to_memory
    - If they're searching for specific information, use search_memory
    - If they want to see everything stored, use get_all_memory""",
    tools=[add_to_memory, search_memory, get_all_memory],
)

"""
### 6. Implement the Main Runtime Loop
"""
logger.info("### 6. Implement the Main Runtime Loop")

async def main():
    current_agent: Agent[Mem0Context] = memory_agent
    input_items: list[TResponseInputItem] = []
    context = Mem0Context()
    while True:
        user_input = input("Enter your message (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        input_items.append({"content": user_input, "role": "user"})
        async def run_async_code_0b5cb604():
            async def run_async_code_7b04ea93():
                result = await Runner.run(current_agent, input_items, context=context)
                return result
            result = asyncio.run(run_async_code_7b04ea93())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_0b5cb604())
        logger.success(format_json(result))
        for new_item in result.new_items:
            agent_name = new_item.agent.name
            if isinstance(new_item, MessageOutputItem):
                logger.debug(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
            elif isinstance(new_item, ToolCallItem):
                logger.debug(f"{agent_name}: Calling a tool")
            elif isinstance(new_item, ToolCallOutputItem):
                logger.debug(f"{agent_name}: Tool call output: {new_item.output}")
            else:
                logger.debug(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")
        input_items = result.to_input_list()

if __name__ == "__main__":
    asyncio.run(main())

"""
## Usage Examples

### Storing Information

User: Remember that my favorite color is blue
Agent: Calling a tool
Agent: Tool call output: Stored message: my favorite color is blue
Agent: I've stored that your favorite color is blue in my memory. I'll remember that for future conversations.

### Searching Memory

User: What's my favorite color?
Agent: Calling a tool
Agent: Tool call output: my favorite color is blue
Agent: Your favorite color is blue, based on what you've told me earlier.

### Retrieving All Memories

User: What do you know about me?
Agent: Calling a tool
Agent: Tool call output: favorite color is blue
my birthday is on March 15
Agent: Based on our previous conversations, I know that:
1. Your favorite color is blue
2. Your birthday is on March 15

## Advanced Configuration

### Custom User IDs

You can specify different user IDs to maintain separate memory stores for multiple users:
"""
logger.info("## Usage Examples")

context = Mem0Context(user_id="user123")

"""
## Resources

- [Mem0 Documentation](https://docs.mem0.ai)
- [Mem0 Dashboard](https://app.mem0.ai/dashboard)
- [API Reference](https://docs.mem0.ai/api-reference)
"""
logger.info("## Resources")

logger.info("\n\n[DONE]", bright=True)