import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import AsyncMemoryClient
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
title: Async Client
description: 'Asynchronous client for Mem0'
icon: "bolt"
iconType: "solid"
---

The `AsyncMemoryClient` is an asynchronous client for interacting with the Mem0 API. It provides similar functionality to the synchronous `MemoryClient` but allows for non-blocking operations, which can be beneficial in applications that require high concurrency.

## Initialization

To use the async client, you first need to initialize it:

<CodeGroup>
"""
logger.info("## Initialization")


os.environ["MEM0_API_KEY"] = "your-api-key"

client = AsyncMemoryClient()

"""

"""

{ MemoryClient } = require('mem0ai')
client = new MemoryClient({ apiKey: 'your-api-key'})

"""
</CodeGroup>

## Methods

The `AsyncMemoryClient` provides the following methods:

### Add

Add a new memory asynchronously.

<CodeGroup>
"""
logger.info("## Methods")

messages = [
    {"role": "user", "content": "Alice loves playing badminton"},
    {"role": "assistant", "content": "That's great! Alice is a fitness freak"},
]
async def run_async_code_dd74c990():
    await client.add(messages, user_id="alice")
    return 
 = asyncio.run(run_async_code_dd74c990())
logger.success(format_json())

"""

"""

messages = [
    {"role": "user", "content": "Alice loves playing badminton"},
    {"role": "assistant", "content": "That's great! Alice is a fitness freak"},
]
async def run_async_code_0b3e9bea():
    await client.add(messages, { user_id: "alice" })
    return 
 = asyncio.run(run_async_code_0b3e9bea())
logger.success(format_json())

"""
</CodeGroup>

### Search

Search for memories based on a query asynchronously.

<CodeGroup>
"""
logger.info("### Search")

async def run_async_code_ab40107c():
    await client.search("What is Alice's favorite sport?", user_id="alice")
    return 
 = asyncio.run(run_async_code_ab40107c())
logger.success(format_json())

"""

"""

async def run_async_code_c7c3b9e8():
    await client.search("What is Alice's favorite sport?", { user_id: "alice" })
    return 
 = asyncio.run(run_async_code_c7c3b9e8())
logger.success(format_json())

"""
</CodeGroup>

### Get All

Retrieve all memories for a user asynchronously.

<CodeGroup>
"""
logger.info("### Get All")

async def run_async_code_5cf7426a():
    await client.get_all(user_id="alice")
    return 
 = asyncio.run(run_async_code_5cf7426a())
logger.success(format_json())

"""

"""

async def run_async_code_d624cdd8():
    await client.getAll({ user_id: "alice" })
    return 
 = asyncio.run(run_async_code_d624cdd8())
logger.success(format_json())

"""
</CodeGroup>

### Delete

Delete a specific memory asynchronously.

<CodeGroup>
"""
logger.info("### Delete")

async def run_async_code_142ef0c5():
    await client.delete(memory_id="memory-id-here")
    return 
 = asyncio.run(run_async_code_142ef0c5())
logger.success(format_json())

"""

"""

async def run_async_code_95f5548f():
    await client.delete("memory-id-here")
    return 
 = asyncio.run(run_async_code_95f5548f())
logger.success(format_json())

"""
</CodeGroup>

### Delete All

Delete all memories for a user asynchronously.

<CodeGroup>
"""
logger.info("### Delete All")

async def run_async_code_8ae257f6():
    await client.delete_all(user_id="alice")
    return 
 = asyncio.run(run_async_code_8ae257f6())
logger.success(format_json())

"""

"""

async def run_async_code_8d08aa28():
    await client.deleteAll({ user_id: "alice" })
    return 
 = asyncio.run(run_async_code_8d08aa28())
logger.success(format_json())

"""
</CodeGroup>

### History

Get the history of a specific memory asynchronously.

<CodeGroup>
"""
logger.info("### History")

async def run_async_code_233cb257():
    await client.history(memory_id="memory-id-here")
    return 
 = asyncio.run(run_async_code_233cb257())
logger.success(format_json())

"""

"""

async def run_async_code_47ba849b():
    await client.history("memory-id-here")
    return 
 = asyncio.run(run_async_code_47ba849b())
logger.success(format_json())

"""
</CodeGroup>

### Users

Get all users, agents, and runs which have memories associated with them asynchronously.

<CodeGroup>
"""
logger.info("### Users")

async def run_async_code_829919a6():
    await client.users()
    return 
 = asyncio.run(run_async_code_829919a6())
logger.success(format_json())

"""

"""

async def run_async_code_829919a6():
    await client.users()
    return 
 = asyncio.run(run_async_code_829919a6())
logger.success(format_json())

"""
</CodeGroup>

### Reset

Reset the client, deleting all users and memories asynchronously.

<CodeGroup>
"""
logger.info("### Reset")

async def run_async_code_62c1dacf():
    await client.reset()
    return 
 = asyncio.run(run_async_code_62c1dacf())
logger.success(format_json())

"""

"""

async def run_async_code_62c1dacf():
    await client.reset()
    return 
 = asyncio.run(run_async_code_62c1dacf())
logger.success(format_json())

"""
</CodeGroup>

## Conclusion

The `AsyncMemoryClient` provides a powerful way to interact with the Mem0 API asynchronously, allowing for more efficient and responsive applications. By using this client, you can perform memory operations without blocking your application's execution.

If you have any questions or need further assistance, please don't hesitate to reach out:

<Snippet file="get-help.mdx" />
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)