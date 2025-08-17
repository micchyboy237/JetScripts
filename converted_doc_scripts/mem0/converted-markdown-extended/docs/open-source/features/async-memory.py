import asyncio
from jet.transformers.formatters import format_json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from functools import wraps
from jet.logger import CustomLogger
from mem0 import AsyncMemory
from mem0.configs.base import MemoryConfig
from openai import AsyncMLX
import asyncio
import logging
import os
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Async Memory
description: 'Asynchronous memory for Mem0'
icon: "bolt"
iconType: "solid"
---

## AsyncMemory

The `AsyncMemory` class is a direct asynchronous interface to Mem0's in-process memory operations. Unlike the memory, which interacts with an API, `AsyncMemory` works directly with the underlying storage systems. This makes it ideal for applications where you want to embed Mem0 directly into your codebase.

### Initialization

To use `AsyncMemory`, import it from the `mem0.memory` module:
"""
logger.info("## AsyncMemory")


memory = AsyncMemory()

custom_config = MemoryConfig(
)
memory = AsyncMemory(config=custom_config)

"""
### Key Features

1. **Non-blocking Operations** - All memory operations use `asyncio` to avoid blocking the event loop
2. **Concurrent Processing** - Parallel execution of vector store and graph operations
3. **Efficient Resource Utilization** - Better handling of I/O bound operations
4. **Compatible with Async Frameworks** - Seamless integration with FastAPI, aiohttp, and other async frameworks

### Methods

All methods in `AsyncMemory` have the same parameters as the synchronous `Memory` class but are designed to be used with `async/await`.

#### Create memories

Add a new memory asynchronously:
"""
logger.info("### Key Features")

try:
    async def async_func_1():
        result = await memory.add(
            messages=[
                {"role": "user", "content": "I'm travelling to SF"},
                {"role": "assistant", "content": "That's great to hear!"}
            ],
            user_id="alice"
        )
        return result
    result = asyncio.run(async_func_1())
    logger.success(format_json(result))
    logger.debug("Memory added successfully:", result)
except Exception as e:
    logger.debug(f"Error adding memory: {e}")

"""
#### Retrieve memories

Retrieve memories related to a query:
"""
logger.info("#### Retrieve memories")

try:
    async def async_func_1():
        results = await memory.search(
            query="Where am I travelling?",
            user_id="alice"
        )
        return results
    results = asyncio.run(async_func_1())
    logger.success(format_json(results))
    logger.debug("Found memories:", results)
except Exception as e:
    logger.debug(f"Error searching memories: {e}")

"""
#### List memories

List all memories for a `user_id`, `agent_id`, and/or `run_id`:
"""
logger.info("#### List memories")

try:
    async def run_async_code_5f0de0ee():
        async def run_async_code_5ca4ab92():
            all_memories = await memory.get_all(user_id="alice")
            return all_memories
        all_memories = asyncio.run(run_async_code_5ca4ab92())
        logger.success(format_json(all_memories))
        return all_memories
    all_memories = asyncio.run(run_async_code_5f0de0ee())
    logger.success(format_json(all_memories))
    logger.debug(f"Retrieved {len(all_memories)} memories")
except Exception as e:
    logger.debug(f"Error retrieving memories: {e}")

"""
#### Get specific memory

Retrieve a specific memory by its ID:
"""
logger.info("#### Get specific memory")

try:
    async def run_async_code_110b4d1f():
        async def run_async_code_4d9e42f6():
            specific_memory = await memory.get(memory_id="memory-id-here")
            return specific_memory
        specific_memory = asyncio.run(run_async_code_4d9e42f6())
        logger.success(format_json(specific_memory))
        return specific_memory
    specific_memory = asyncio.run(run_async_code_110b4d1f())
    logger.success(format_json(specific_memory))
    logger.debug("Retrieved memory:", specific_memory)
except Exception as e:
    logger.debug(f"Error retrieving memory: {e}")

"""
#### Update memory

Update an existing memory by ID:
"""
logger.info("#### Update memory")

try:
    async def async_func_1():
        updated_memory = await memory.update(
            memory_id="memory-id-here",
            data="I'm travelling to Seattle"
        )
        return updated_memory
    updated_memory = asyncio.run(async_func_1())
    logger.success(format_json(updated_memory))
    logger.debug("Memory updated successfully:", updated_memory)
except Exception as e:
    logger.debug(f"Error updating memory: {e}")

"""
#### Delete memory

Delete a specific memory by ID:
"""
logger.info("#### Delete memory")

try:
    async def run_async_code_d6286070():
        async def run_async_code_fcba09c4():
            result = await memory.delete(memory_id="memory-id-here")
            return result
        result = asyncio.run(run_async_code_fcba09c4())
        logger.success(format_json(result))
        return result
    result = asyncio.run(run_async_code_d6286070())
    logger.success(format_json(result))
    logger.debug("Memory deleted successfully")
except Exception as e:
    logger.debug(f"Error deleting memory: {e}")

"""
#### Delete all memories

Delete all memories for a specific user, agent, or run:
"""
logger.info("#### Delete all memories")

try:
    async def run_async_code_74952015():
        async def run_async_code_717a298d():
            result = await memory.delete_all(user_id="alice")
            return result
        result = asyncio.run(run_async_code_717a298d())
        logger.success(format_json(result))
        return result
    result = asyncio.run(run_async_code_74952015())
    logger.success(format_json(result))
    logger.debug("All memories deleted successfully")
except Exception as e:
    logger.debug(f"Error deleting memories: {e}")

"""
<Note>
At least one filter (user_id, agent_id, or run_id) is required when using delete_all.
</Note>

### Advanced Memory Organization

AsyncMemory supports the same three-parameter organization system as the synchronous Memory class:
"""
logger.info("### Advanced Memory Organization")

await memory.add(
    messages=[{"role": "user", "content": "I prefer vegetarian food"}],
    user_id="alice",
    agent_id="diet-assistant",
    run_id="consultation-001"
)

async def run_async_code_9c775229():
    async def run_async_code_98d0f653():
        all_user_memories = await memory.get_all(user_id="alice")
        return all_user_memories
    all_user_memories = asyncio.run(run_async_code_98d0f653())
    logger.success(format_json(all_user_memories))
    return all_user_memories
all_user_memories = asyncio.run(run_async_code_9c775229())
logger.success(format_json(all_user_memories))
async def run_async_code_26eabed5():
    async def run_async_code_86919a53():
        agent_memories = await memory.get_all(user_id="alice", agent_id="diet-assistant")
        return agent_memories
    agent_memories = asyncio.run(run_async_code_86919a53())
    logger.success(format_json(agent_memories))
    return agent_memories
agent_memories = asyncio.run(run_async_code_26eabed5())
logger.success(format_json(agent_memories))
async def run_async_code_5929ebe7():
    async def run_async_code_a7910d82():
        session_memories = await memory.get_all(user_id="alice", run_id="consultation-001")
        return session_memories
    session_memories = asyncio.run(run_async_code_a7910d82())
    logger.success(format_json(session_memories))
    return session_memories
session_memories = asyncio.run(run_async_code_5929ebe7())
logger.success(format_json(session_memories))
async def async_func_10():
    specific_memories = await memory.get_all(
        user_id="alice",
        agent_id="diet-assistant",
        run_id="consultation-001"
    )
    return specific_memories
specific_memories = asyncio.run(async_func_10())
logger.success(format_json(specific_memories))

async def run_async_code_daf94f16():
    async def run_async_code_46b35b9b():
        general_search = await memory.search("What do you know about me?", user_id="alice")
        return general_search
    general_search = asyncio.run(run_async_code_46b35b9b())
    logger.success(format_json(general_search))
    return general_search
general_search = asyncio.run(run_async_code_daf94f16())
logger.success(format_json(general_search))
async def run_async_code_f9946d5d():
    async def run_async_code_245c8d50():
        agent_search = await memory.search("What do you know about me?", user_id="alice", agent_id="diet-assistant")
        return agent_search
    agent_search = asyncio.run(run_async_code_245c8d50())
    logger.success(format_json(agent_search))
    return agent_search
agent_search = asyncio.run(run_async_code_f9946d5d())
logger.success(format_json(agent_search))
async def run_async_code_ffca7433():
    async def run_async_code_827d8435():
        session_search = await memory.search("What do you know about me?", user_id="alice", run_id="consultation-001")
        return session_search
    session_search = asyncio.run(run_async_code_827d8435())
    logger.success(format_json(session_search))
    return session_search
session_search = asyncio.run(run_async_code_ffca7433())
logger.success(format_json(session_search))

"""
#### Memory History

Get the history of changes for a specific memory:
"""
logger.info("#### Memory History")

try:
    async def run_async_code_8f34210e():
        async def run_async_code_c35797a6():
            history = await memory.history(memory_id="memory-id-here")
            return history
        history = asyncio.run(run_async_code_c35797a6())
        logger.success(format_json(history))
        return history
    history = asyncio.run(run_async_code_8f34210e())
    logger.success(format_json(history))
    logger.debug("Memory history:", history)
except Exception as e:
    logger.debug(f"Error retrieving history: {e}")

"""
### Example: Concurrent Usage with Other APIs

`AsyncMemory` can be effectively combined with other async operations. Here's an example showing how to use it alongside MLX API calls in separate threads:
"""
logger.info("### Example: Concurrent Usage with Other APIs")


async_openai_client = AsyncMLX()
async_memory = AsyncMemory()

async def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    try:
        async def run_async_code_3cf4cc92():
            async def run_async_code_d691266a():
                search_result = await async_memory.search(query=message, user_id=user_id, limit=3)
                return search_result
            search_result = asyncio.run(run_async_code_d691266a())
            logger.success(format_json(search_result))
            return search_result
        search_result = asyncio.run(run_async_code_3cf4cc92())
        logger.success(format_json(search_result))
        relevant_memories = search_result["results"]
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)

        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
        async def run_async_code_20dfc597():
            async def run_async_code_26264579():
                response = await async_openai_client.chat.completions.create(model="llama-3.2-3b-instruct", messages=messages)
                return response
            response = asyncio.run(run_async_code_26264579())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_20dfc597())
        logger.success(format_json(response))
        assistant_response = response.choices[0].message.content

        messages.append({"role": "assistant", "content": assistant_response})
        async def run_async_code_fd0c2a10():
            await async_memory.add(messages, user_id=user_id)
            return 
         = asyncio.run(run_async_code_fd0c2a10())
        logger.success(format_json())

        return assistant_response
    except Exception as e:
        logger.debug(f"Error in chat_with_memories: {e}")
        return "I apologize, but I encountered an error processing your request."

async def async_main():
    logger.debug("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            logger.debug("Goodbye!")
            break
        async def run_async_code_7fd56ac1():
            async def run_async_code_c08a0570():
                response = await chat_with_memories(user_input)
                return response
            response = asyncio.run(run_async_code_c08a0570())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_7fd56ac1())
        logger.success(format_json(response))
        logger.debug(f"AI: {response}")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()

"""
## Error Handling and Best Practices

### Common Error Types

When working with `AsyncMemory`, you may encounter these common errors:

#### Connection and Configuration Errors
"""
logger.info("## Error Handling and Best Practices")


async def handle_initialization_errors():
    try:
        config = MemoryConfig(
            vector_store={"provider": "chroma", "config": {"path": "./chroma_db"}},
            llm={"provider": "openai", "config": {"model": "llama-3.2-3b-instruct"}}
        )
        memory = AsyncMemory(config=config)
        logger.debug("AsyncMemory initialized successfully")
    except ValueError as e:
        logger.debug(f"Configuration error: {e}")
    except ConnectionError as e:
        logger.debug(f"Connection error: {e}")
    except Exception as e:
        logger.debug(f"Unexpected initialization error: {e}")

asyncio.run(handle_initialization_errors())

"""
#### Memory Operation Errors
"""
logger.info("#### Memory Operation Errors")

async def handle_memory_operation_errors():
    memory = AsyncMemory()

    try:
        async def run_async_code_79b05454():
            async def run_async_code_0ed41d2a():
                result = await memory.get(memory_id="non-existent-id")
                return result
            result = asyncio.run(run_async_code_0ed41d2a())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_79b05454())
        logger.success(format_json(result))
    except ValueError as e:
        logger.debug(f"Invalid memory ID: {e}")
    except Exception as e:
        logger.debug(f"Memory retrieval error: {e}")

    try:
        async def run_async_code_b7a78e7d():
            async def run_async_code_d6966032():
                results = await memory.search(query="", user_id="alice")
                return results
            results = asyncio.run(run_async_code_d6966032())
            logger.success(format_json(results))
            return results
        results = asyncio.run(run_async_code_b7a78e7d())
        logger.success(format_json(results))
    except ValueError as e:
        logger.debug(f"Invalid search query: {e}")
    except Exception as e:
        logger.debug(f"Search error: {e}")

"""
### Performance Optimization

#### Concurrent Operations

Take advantage of AsyncMemory's concurrent capabilities:
"""
logger.info("### Performance Optimization")

async def batch_operations():
    memory = AsyncMemory()

    tasks = []
    for i in range(5):
        task = memory.add(
            messages=[{"role": "user", "content": f"Message {i}"}],
            user_id=f"user_{i}"
        )
        tasks.append(task)

    try:
        async def run_async_code_e3ccfa0f():
            async def run_async_code_eb4f5cfd():
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            results = asyncio.run(run_async_code_eb4f5cfd())
            logger.success(format_json(results))
            return results
        results = asyncio.run(run_async_code_e3ccfa0f())
        logger.success(format_json(results))
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.debug(f"Task {i} failed: {result}")
            else:
                logger.debug(f"Task {i} completed successfully")
    except Exception as e:
        logger.debug(f"Batch operation error: {e}")

"""
#### Resource Management

Properly manage AsyncMemory lifecycle:
"""
logger.info("#### Resource Management")


@asynccontextmanager
async def get_memory():
    memory = AsyncMemory()
    try:
        yield memory
    finally:
        pass

async def safe_memory_usage():
    async def async_func_12():
        async with get_memory() as memory:
            try:
                async def run_async_code_ddf76ca0():
                    result = await memory.search("test query", user_id="alice")
                    return result
                result = asyncio.run(run_async_code_ddf76ca0())
                logger.success(format_json(result))
                return result
            except Exception as e:
                logger.debug(f"Memory operation failed: {e}")
                return None
        return result

    result = asyncio.run(async_func_12())
    logger.success(format_json(result))

"""
### Timeout and Retry Strategies

Implement timeout and retry logic for robustness:
"""
logger.info("### Timeout and Retry Strategies")

async def with_timeout_and_retry(operation, max_retries=3, timeout=10.0):
    for attempt in range(max_retries):
        try:
            async def run_async_code_0e134542():
                async def run_async_code_78569582():
                    result = await asyncio.wait_for(operation(), timeout=timeout)
                    return result
                result = asyncio.run(run_async_code_78569582())
                logger.success(format_json(result))
                return result
            result = asyncio.run(run_async_code_0e134542())
            logger.success(format_json(result))
            return result
        except asyncio.TimeoutError:
            logger.debug(f"Timeout on attempt {attempt + 1}")
        except Exception as e:
            logger.debug(f"Error on attempt {attempt + 1}: {e}")

        if attempt < max_retries - 1:
            async def run_async_code_1a82d117():
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                return 
             = asyncio.run(run_async_code_1a82d117())
            logger.success(format_json())

    raise Exception(f"Operation failed after {max_retries} attempts")

async def robust_memory_search():
    memory = AsyncMemory()

    async def search_operation():
        async def run_async_code_ebf93cb7():
            return await memory.search("test query", user_id="alice")
            return 
         = asyncio.run(run_async_code_ebf93cb7())
        logger.success(format_json())

    try:
        async def run_async_code_8d05c788():
            async def run_async_code_3b87b373():
                result = await with_timeout_and_retry(search_operation)
                return result
            result = asyncio.run(run_async_code_3b87b373())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_8d05c788())
        logger.success(format_json(result))
        logger.debug("Search successful:", result)
    except Exception as e:
        logger.debug(f"Search failed permanently: {e}")

"""
### Integration with Async Frameworks

#### FastAPI Integration
"""
logger.info("### Integration with Async Frameworks")


app = FastAPI()
memory = AsyncMemory()

@app.post("/memories/")
async def add_memory(messages: list, user_id: str):
    try:
        async def run_async_code_f05c46d1():
            async def run_async_code_454946a5():
                result = await memory.add(messages=messages, user_id=user_id)
                return result
            result = asyncio.run(run_async_code_454946a5())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_f05c46d1())
        logger.success(format_json(result))
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/search")
async def search_memories(query: str, user_id: str, limit: int = 10):
    try:
        async def run_async_code_70515dc3():
            async def run_async_code_041d81f0():
                result = await memory.search(query=query, user_id=user_id, limit=limit)
                return result
            result = asyncio.run(run_async_code_041d81f0())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_70515dc3())
        logger.success(format_json(result))
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
### Troubleshooting Guide

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| **Initialization fails** | Missing dependencies, invalid config | Check dependencies, validate configuration |
| **Slow operations** | Large datasets, network latency | Implement caching, optimize queries |
| **Memory not found** | Invalid memory ID, deleted memory | Validate IDs, implement existence checks |
| **Connection timeouts** | Network issues, server overload | Implement retry logic, check network |
| **Out of memory errors** | Large batch operations | Process in smaller batches |

### Monitoring and Logging

Add comprehensive logging to your async memory operations:
"""
logger.info("### Troubleshooting Guide")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_async_operation(operation_name):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {operation_name}")
            try:
                async def run_async_code_49fdabc0():
                    async def run_async_code_319f0b33():
                        result = await func(*args, **kwargs)
                        return result
                    result = asyncio.run(run_async_code_319f0b33())
                    logger.success(format_json(result))
                    return result
                result = asyncio.run(run_async_code_49fdabc0())
                logger.success(format_json(result))
                duration = time.time() - start_time
                logger.info(f"{operation_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_name} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

@log_async_operation("Memory Add")
async def logged_memory_add(memory, messages, user_id):
    async def run_async_code_0778b449():
        return await memory.add(messages=messages, user_id=user_id)
        return 
     = asyncio.run(run_async_code_0778b449())
    logger.success(format_json())

"""
If you have any questions or need further assistance, please don't hesitate to reach out:

<Snippet file="get-help.mdx" />
"""
logger.info("If you have any questions or need further assistance, please don't hesitate to reach out:")

logger.info("\n\n[DONE]", bright=True)