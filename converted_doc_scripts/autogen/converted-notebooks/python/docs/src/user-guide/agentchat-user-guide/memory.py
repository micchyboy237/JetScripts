import asyncio

import redis
from jet.llm.mlx.memory import MemoryManager
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.memory.mem0 import Mem0Memory
from autogen_ext.memory.redis import RedisMemory, RedisMemoryConfig
from jet.llm.mlx.autogen_ext.mlx_chat_completion_client import MLXChatCompletionClient
from jet.logger import CustomLogger
from logging import WARNING
from pathlib import Path
from typing import List
import aiofiles
import aiohttp
import os
import re
import shutil
import tempfile


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## Memory and RAG

There are several use cases where it is valuable to maintain a _store_ of useful facts that can be intelligently added to the context of the agent just before a specific step. The typically use case here is a RAG pattern where a query is used to retrieve relevant information from a database that is then added to the agent's context.


AgentChat provides a {py:class}`~autogen_core.memory.Memory` protocol that can be extended to provide this functionality.  The key methods are `query`, `update_context`,  `add`, `clear`, and `close`. 

- `add`: add new entries to the memory store
- `query`: retrieve relevant information from the memory store 
- `update_context`: mutate an agent's internal `model_context` by adding the retrieved information (used in the {py:class}`~autogen_agentchat.agents.AssistantAgent` class) 
- `clear`: clear all entries from the memory store
- `close`: clean up any resources used by the memory store  


## ListMemory Example

{py:class}`~autogen_core.memory.ListMemory` is provided as an example implementation of the {py:class}`~autogen_core.memory.Memory` protocol. It is a simple list-based memory implementation that maintains memories in chronological order, appending the most recent memories to the model's context. The implementation is designed to be straightforward and predictable, making it easy to understand and debug.
In the following example, we will use ListMemory to maintain a memory bank of user preferences and demonstrate how it can be used to provide consistent context for agent responses over time.
"""
logger.info("## Memory and RAG")


user_memory = ListMemory()


async def run_async_code_daf84d7d():
    await user_memory.add(MemoryContent(content="The weather should be in metric units", mime_type=MemoryMimeType.TEXT))
    return
asyncio.run(run_async_code_daf84d7d())


async def run_async_code_4e246ca5():
    await user_memory.add(MemoryContent(content="Meal recipe must be vegan", mime_type=MemoryMimeType.TEXT))
    return
asyncio.run(run_async_code_4e246ca5())


async def get_weather(city: str, units: str = "imperial") -> str:
    if units == "imperial":
        return f"The weather in {city} is 73 °F and Sunny."
    elif units == "metric":
        return f"The weather in {city} is 23 °C and Sunny."
    else:
        return f"Sorry, I don't know the weather in {city}."


assistant_agent = AssistantAgent(
    name="assistant_agent",
    model_client=MLXChatCompletionClient(
        model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
    ),
    tools=[get_weather],
    memory=[user_memory],
)

stream = assistant_agent.run_stream(task="What is the weather in New York?")


async def run_async_code_71db6073():
    await Console(stream)
    return
asyncio.run(run_async_code_71db6073())


"""
We can inspect that the `assistant_agent` model_context is actually updated with the retrieved memory entries.  The `transform` method is used to format the retrieved memory entries into a string that can be used by the agent.  In this case, we simply concatenate the content of each memory entry into a single string.
"""
logger.info("We can inspect that the `assistant_agent` model_context is actually updated with the retrieved memory entries.  The `transform` method is used to format the retrieved memory entries into a string that can be used by the agent.  In this case, we simply concatenate the content of each memory entry into a single string.")


async def run_async_code_16b72f9b():
    await assistant_agent._model_context.get_messages()
    return
asyncio.run(run_async_code_16b72f9b())


"""
We see above that the weather is returned in Centigrade as stated in the user preferences. 

Similarly, assuming we ask a separate question about generating a meal plan, the agent is able to retrieve relevant information from the memory store and provide a personalized (vegan) response.
"""
logger.info(
    "We see above that the weather is returned in Centigrade as stated in the user preferences.")

stream = assistant_agent.run_stream(task="Write brief meal recipe with broth")


async def run_async_code_71db6073():
    await Console(stream)
    return
asyncio.run(run_async_code_71db6073())


"""
## Custom Memory Stores (Vector DBs, etc.)

You can build on the `Memory` protocol to implement more complex memory stores. For example, you could implement a custom memory store that uses a vector database to store and retrieve information, or a memory store that uses a machine learning model to generate personalized responses based on the user's preferences etc.

Specifically, you will need to overload the `add`, `query` and `update_context`  methods to implement the desired functionality and pass the memory store to your agent.


Currently the following example memory stores are available as part of the {py:class}`~autogen_ext` extensions package.

- `autogen_ext.memory.chromadb.ChromaDBVectorMemory`: A memory store that uses a vector database to store and retrieve information.

- `autogen_ext.memory.chromadb.SentenceTransformerEmbeddingFunctionConfig`: A configuration class for the SentenceTransformer embedding function used by the `ChromaDBVectorMemory` store. Note that other embedding functions such as `autogen_ext.memory.openai.MLXEmbeddingFunctionConfig` can also be used with the `ChromaDBVectorMemory` store.

- `autogen_ext.memory.redis.RedisMemory`: A memory store that uses a Redis vector database to store and retrieve information.
"""
logger.info("## Custom Memory Stores (Vector DBs, etc.)")


with tempfile.TemporaryDirectory() as tmpdir:
    chroma_user_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="preferences",
            persistence_path=tmpdir,  # Use the temp directory here
            k=2,  # Return top k results
            score_threshold=0.4,  # Minimum similarity score
            embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                model_name="all-MiniLM-L6-v2"  # Use default model for testing
            ),
        )
    )

    async def run_async_code_1a2b3c4d():
        await chroma_user_memory.add(
            MemoryContent(
                content="The weather should be in metric units",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "units"},
            )
        )
        await chroma_user_memory.add(
            MemoryContent(
                content="Meal recipe must be vegan",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "dietary"},
            )
        )
        return
    asyncio.run(run_async_code_1a2b3c4d())

    model_client = MLXChatCompletionClient(
        model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
    )

    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        tools=[get_weather],
        memory=[chroma_user_memory],
    )

    stream = assistant_agent.run_stream(
        task="What is the weather in New York?")

    async def run_async_code_8cdf6b5b():
        await Console(stream)
        return
    asyncio.run(run_async_code_8cdf6b5b())

    async def run_async_code_3902376f():
        await model_client.close()
        return
    asyncio.run(run_async_code_3902376f())

    async def run_async_code_5b1f6c7c():
        await chroma_user_memory.close()
        return
    asyncio.run(run_async_code_5b1f6c7c())


"""
Note that you can also serialize the ChromaDBVectorMemory and save it to disk.
"""
logger.info(
    "Note that you can also serialize the ChromaDBVectorMemory and save it to disk.")

chroma_user_memory.dump_component().model_dump_json()

"""
### Redis Memory
You can perform the same persistent memory storage using Redis. Note, you will need to have a running Redis instance to connect to.

See {py:class}`~autogen_ext.memory.redis.RedisMemory` for instructions to run Redis locally or via Docker.
"""
logger.info("### Redis Memory")


# Initialize RedisMemory
logger.debug("Initializing RedisMemory for debugging")
redis_memory = RedisMemory(
    config=RedisMemoryConfig(
        redis_url="redis://localhost:6379/0",  # Explicitly specify DB 0
        index_name="chat_history",
        prefix="memory",
    )
)


async def run_async_code_8a3f1b2c():
    try:
        logger.debug(
            "RedisMemory initialized with redis_url=redis://localhost:6379/0, index_name=chat_history")

        # Verify Redis connection
        client = redis.Redis.from_url(
            "redis://localhost:6379/0", decode_responses=True)
        client.ping()
        logger.debug("Redis connection successful (ping)")

        # Check if chat_history index exists
        indexes = client.execute_command("FT._LIST")
        logger.debug(f"Redis indexes: {indexes}")
        if "chat_history" not in indexes:
            logger.warning("chat_history index not found")

        # Add entries
        logger.debug("Adding 'The weather should be in metric units'")
        await redis_memory.add(
            MemoryContent(
                content="The weather should be in metric units",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "units"},
            )
        )
        logger.debug("Added first entry")

        logger.debug("Adding 'Meal recipe must be vegan'")
        await redis_memory.add(
            MemoryContent(
                content="Meal recipe must be vegan",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "dietary"},
            )
        )
        logger.debug("Added second entry")

        # Verify data immediately after adding
        keys = client.keys("memory*")
        logger.debug(f"Keys found after adding: {keys}")
        for key in keys:
            data = client.hgetall(key)
            logger.debug(f"Data for key {key}: {data}")

        client.close()
        return
    except Exception as e:
        logger.error(f"Error in run_async_code_8a3f1b2c: {str(e)}")
        raise
asyncio.run(run_async_code_8a3f1b2c())

model_client = MLXChatCompletionClient(
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
)

assistant_agent = AssistantAgent(
    name="assistant_agent",
    model_client=model_client,
    tools=[get_weather],
    memory=[redis_memory],
)

stream = assistant_agent.run_stream(task="What is the weather in New York?")


async def run_async_code_71db6073():
    await Console(stream)
    return
asyncio.run(run_async_code_71db6073())


async def run_async_code_0349fda4():
    await model_client.close()
    return
asyncio.run(run_async_code_0349fda4())


async def run_async_code_952685a8():
    await redis_memory.close()
    return
asyncio.run(run_async_code_952685a8())


"""
## RAG Agent: Putting It All Together

The RAG (Retrieval Augmented Generation) pattern which is common in building AI systems encompasses two distinct phases:

1. **Indexing**: Loading documents, chunking them, and storing them in a vector database
2. **Retrieval**: Finding and using relevant chunks during conversation runtime

In our previous examples, we manually added items to memory and passed them to our agents. In practice, the indexing process is usually automated and based on much larger document sources like product documentation, internal files, or knowledge bases.

> Note: The quality of a RAG system is dependent on the quality of the chunking and retrieval process (models, embeddings, etc.). You may need to experiement with more advanced chunking and retrieval models to get the best results.

### Building a Simple RAG Agent

To begin, let's create a simple document indexer that we will used to load documents, chunk them, and store them in a `ChromaDBVectorMemory` memory store.
"""
logger.info("## RAG Agent: Putting It All Together")


class SimpleDocumentIndexer:
    """Basic document indexer for AutoGen Memory."""

    def __init__(self, memory: Memory, chunk_size: int = 1500) -> None:
        self.memory = memory
        self.chunk_size = chunk_size

    async def _fetch_content(self, source: str) -> str:
        """Fetch content from URL or file."""
        if source.startswith(("http://", "https://")):
            async with aiohttp.ClientSession() as session:
                async with session.get(source) as response:
                    content = await response.text()
                    logger.success(format_json(content))
                    return content
        else:
            async with aiofiles.open(source, "r", encoding="utf-8") as f:
                content = await f.read()
                logger.success(format_json(content))
                return content

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks: list[str] = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i: i + self.chunk_size]
            chunks.append(chunk.strip())
        return chunks

    async def index_documents(self, sources: List[str]) -> int:
        """Index documents into memory."""
        total_chunks = 0

        for source in sources:
            try:
                content = await self._fetch_content(source)
                logger.success(format_json(content))

                if "<" in content and ">" in content:
                    content = self._strip_html(content)

                chunks = self._split_text(content)

                for i, chunk in enumerate(chunks):
                    await self.memory.add(
                        MemoryContent(
                            content=chunk, mime_type=MemoryMimeType.TEXT, metadata={
                                "source": source, "chunk_index": i}
                        )
                    )

                total_chunks += len(chunks)

            except Exception as e:
                logger.debug(f"Error indexing {source}: {str(e)}")

        return total_chunks


"""
Now let's use our indexer with ChromaDBVectorMemory to build a complete RAG agent:
"""
logger.info(
    "Now let's use our indexer with ChromaDBVectorMemory to build a complete RAG agent:")


rag_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="autogen_docs",
        persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
        k=3,  # Return top 3 results
        score_threshold=0.4,  # Minimum similarity score
    )
)


async def run_async_code_aacd10ef():
    await rag_memory.clear()  # Clear existing memory
    return
asyncio.run(run_async_code_aacd10ef())


async def index_autogen_docs() -> None:
    indexer = SimpleDocumentIndexer(memory=rag_memory)
    sources = [
        "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        "https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/agents.html",
        "https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/teams.html",
        "https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/termination.html",
    ]

    chunks: int = await indexer.index_documents(sources)
    logger.success(format_json(chunks))
    logger.debug(
        f"Indexed {chunks} chunks from {len(sources)} AutoGen documents")


async def run_async_code_97c54031():
    await index_autogen_docs()
    return
asyncio.run(run_async_code_97c54031())


rag_assistant = AssistantAgent(
    name="rag_assistant", model_client=MLXChatCompletionClient(model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats"), memory=[rag_memory]
)

stream = rag_assistant.run_stream(task="What is AgentChat?")


async def run_async_code_71db6073():
    await Console(stream)
    return
asyncio.run(run_async_code_71db6073())


async def run_async_code_cc18dab4():
    await rag_memory.close()
    return
asyncio.run(run_async_code_cc18dab4())


"""
This implementation provides a RAG agent that can answer questions based on AutoGen documentation. When a question is asked, the Memory system  retrieves relevant chunks and adds them to the context, enabling the assistant to generate informed responses.

For production systems, you might want to:
1. Implement more sophisticated chunking strategies
2. Add metadata filtering capabilities
3. Customize the retrieval scoring
4. Optimize embedding models for your specific domain

## Mem0Memory Example

`autogen_ext.memory.mem0.Mem0Memory` provides integration with `Mem0.ai`'s memory system. It supports both cloud-based and local backends, offering advanced memory capabilities for agents. The implementation handles proper retrieval and context updating, making it suitable for production environments.

In the following example, we'll demonstrate how to use `Mem0Memory` to maintain persistent memories across conversations:
"""
logger.info("## Mem0Memory Example")


mem0_memory = MemoryManager(
    limit=5,  # Maximum number of memories to retrieve
)


async def run_async_code_4f7b2c1d():
    await mem0_memory.add(
        content="The weather should be in metric units",
        mime_type=MemoryMimeType.TEXT,
        metadata={"category": "preferences", "type": "units"},
    )

    await mem0_memory.add(
        content="Meal recipe must be vegan",
        mime_type=MemoryMimeType.TEXT,
        metadata={"category": "preferences", "type": "dietary"},
    )
    return
asyncio.run(run_async_code_4f7b2c1d())

assistant_agent = AssistantAgent(
    name="assistant_agent",
    model_client=MLXChatCompletionClient(
        model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
    ),
    tools=[get_weather],
    memory=[mem0_memory],
)

stream = assistant_agent.run_stream(task="What are my dietary preferences?")


async def run_async_code_71db6073():
    await Console(stream)
    return
asyncio.run(run_async_code_71db6073())


"""
The example above demonstrates how Mem0Memory can be used with an assistant agent. The memory integration ensures that:

1. All agent interactions are stored in Mem0 for future reference
2. Relevant memories (like user preferences) are automatically retrieved and added to the context
3. The agent can maintain consistent behavior based on stored memories

Mem0Memory is particularly useful for:
- Long-running agent deployments that need persistent memory
- Applications requiring enhanced privacy controls
- Teams wanting unified memory management across agents
- Use cases needing advanced memory filtering and analytics

Just like ChromaDBVectorMemory, you can serialize Mem0Memory configurations:
"""
logger.info("The example above demonstrates how Mem0Memory can be used with an assistant agent. The memory integration ensures that:")

config_json = mem0_memory.dump_component().model_dump_json()
logger.debug(f"Memory config JSON: {config_json[:100]}...")

logger.info("\n\n[DONE]", bright=True)
