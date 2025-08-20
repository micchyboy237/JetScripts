import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import (
StaticMemoryBlock,
FactExtractionMemoryBlock,
VectorMemoryBlock,
)
from llama_index.core.memory import Memory
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Memory in LlamaIndex

The `Memory` class in LlamaIndex is used to store and retrieve both short-term and long-term memory.

You can use it on its own and orchestrate within a custom workflow, or use it within an existing agent.

By default, short-term memory is represented as a FIFO queue of `ChatMessage` objects. Once the queue exceeds a certain size, the last X messages within a flush size are archived and optionally flushed to long-term memory blocks.

Long-term memory is represented as `Memory Block` objects. These objects receive the messages that are flushed from short-term memory, and optionally process them to extract information. Then when memory is retrieved, the short-term and long-term memories are merged together.

## Setup

This notebook will use `MLX` as an LLM/embedding model for various parts of the example.

For vector retrieval, we will rely on `Chroma` as a vector store.
"""
logger.info("# Memory in LlamaIndex")

# %pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-chroma


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

"""
## Short-term Memory

Let's explore how to configure various components of short-term memory.

For visual purposes, we will set some low token limits to more easily observe the memory behavior.
"""
logger.info("## Short-term Memory")


memory = Memory.from_defaults(
    session_id="my_session",
    token_limit=50,  # Normally you would set this to be closer to the LLM context window (i.e. 75,000, etc.)
    token_flush_size=10,
    chat_history_token_ratio=0.7,
)

"""
Let's review the configuration we used and what it means:

- `session_id`: A unique identifier for the session. Used to mark chat messages in a SQL database as belonging to a specific session.
- `token_limit`: The maximum number of tokens that can be stored in short-term + long-term memory.
- `chat_history_token_ratio`: The ratio of tokens in the short-term chat history to the total token limit. Here this means that 50*0.7 = 35 tokens are allocated to short-term memory, and the rest is allocated to long-term memory.
- `token_flush_size`: The number of tokens to flush to long-term memory when the token limit is exceeded. Note that we did not configure long-term memory, so these messages are merely archived in the database and removed from the short-term memory.

Using our memory, we can manually add some messages and observe how it works.
"""
logger.info("Let's review the configuration we used and what it means:")


for i in range(100):
    await memory.aput_messages(
        [
            ChatMessage(role="user", content="Hello, world!"),
            ChatMessage(role="assistant", content="Hello, world to you too!"),
            ChatMessage(role="user", content="What is the capital of France?"),
            ChatMessage(
                role="assistant", content="The capital of France is Paris."
            ),
        ]
    )

"""
Since our token limit is small, we will only see the last 4 messages in short-term memory (since this fits withint the 50*0.7 limit)
"""
logger.info("Since our token limit is small, we will only see the last 4 messages in short-term memory (since this fits withint the 50*0.7 limit)")

async def run_async_code_b0fb598e():
    async def run_async_code_b24d0f84():
        current_chat_history = await memory.aget()
        return current_chat_history
    current_chat_history = asyncio.run(run_async_code_b24d0f84())
    logger.success(format_json(current_chat_history))
    return current_chat_history
current_chat_history = asyncio.run(run_async_code_b0fb598e())
logger.success(format_json(current_chat_history))
for msg in current_chat_history:
    logger.debug(msg)

"""
If we retrieva all messages, we will find all 400 messages.
"""
logger.info("If we retrieva all messages, we will find all 400 messages.")

async def run_async_code_447759e9():
    async def run_async_code_8c9bda0f():
        all_messages = await memory.aget_all()
        return all_messages
    all_messages = asyncio.run(run_async_code_8c9bda0f())
    logger.success(format_json(all_messages))
    return all_messages
all_messages = asyncio.run(run_async_code_447759e9())
logger.success(format_json(all_messages))
logger.debug(len(all_messages))

"""
We can clear the memory at any time to start fresh.
"""
logger.info("We can clear the memory at any time to start fresh.")

async def run_async_code_d53bf12e():
    await memory.areset()
    return 
 = asyncio.run(run_async_code_d53bf12e())
logger.success(format_json())

async def run_async_code_447759e9():
    async def run_async_code_8c9bda0f():
        all_messages = await memory.aget_all()
        return all_messages
    all_messages = asyncio.run(run_async_code_8c9bda0f())
    logger.success(format_json(all_messages))
    return all_messages
all_messages = asyncio.run(run_async_code_447759e9())
logger.success(format_json(all_messages))
logger.debug(len(all_messages))

"""
## Long-term Memory

Long-term memory is represented as `Memory Block` objects. These objects receive the messages that are flushed from short-term memory, and optionally process them to extract information. Then when memory is retrieved, the short-term and long-term memories are merged together.

LlamaIndex provides 3 prebuilt memory blocks:

- `StaticMemoryBlock`: A memory block that stores a static piece of information.
- `FactExtractionMemoryBlock`: A memory block that extracts facts from the chat history.
- `VectorMemoryBlock`: A memory block that stores and retrieves batches of chat messages from a vector database.

Each block has a `priority` that is used when the long-term memory + short-term memory exceeds the token limit. Priority 0 means the block will always be kept in memory, priority 1 means the block will be temporarily disabled, and so on.
"""
logger.info("## Long-term Memory")


llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
embed_model = MLXEmbedding(model="mxbai-embed-large")

client = chromadb.EphemeralClient()
vector_store = ChromaVectorStore(
    chroma_collection=client.get_or_create_collection("test_collection")
)

blocks = [
    StaticMemoryBlock(
        name="core_info",
        static_content="My name is Logan, and I live in Saskatoon. I work at LlamaIndex.",
        priority=0,
    ),
    FactExtractionMemoryBlock(
        name="extracted_info",
        llm=llm,
        max_facts=50,
        priority=1,
    ),
    VectorMemoryBlock(
        name="vector_memory",
        vector_store=vector_store,
        priority=2,
        embed_model=embed_model,
    ),
]

"""
With our blocks created, we can pass them into the `Memory` class.
"""
logger.info("With our blocks created, we can pass them into the `Memory` class.")


memory = Memory.from_defaults(
    session_id="my_session",
    token_limit=30000,
    chat_history_token_ratio=0.02,
    token_flush_size=500,
    memory_blocks=blocks,
    insert_method="user",
)

"""
With this, we can simulate a conversation with an agent and inspect the long-term memory.
"""
logger.info("With this, we can simulate a conversation with an agent and inspect the long-term memory.")


agent = FunctionAgent(
    tools=[],
    llm=llm,
)

user_msgs = [
    "Hi! My name is Logan",
    "What is your opinion on minature shnauzers?",
    "Do they shed a lot?",
    "What breeds are comparable in size?",
    "What is your favorite breed?",
    "Would you recommend owning a dog?",
    "What should I buy to prepare for owning a dog?",
]

for user_msg in user_msgs:
    async def run_async_code_4df6c04c():
        async def run_async_code_e25c1e3e():
            _ = await agent.run(user_msg=user_msg, memory=memory)
            return _
        _ = asyncio.run(run_async_code_e25c1e3e())
        logger.success(format_json(_))
        return _
    _ = asyncio.run(run_async_code_4df6c04c())
    logger.success(format_json(_))

"""
Now, let's inspect the most recent user-message and see what the memory inserts into the user message.

Note that we pass in at least one chat message so that the vector memory actually runs retrieval.
"""
logger.info("Now, let's inspect the most recent user-message and see what the memory inserts into the user message.")

async def run_async_code_657157e4():
    async def run_async_code_eb83d9a2():
        chat_history = await memory.aget()
        return chat_history
    chat_history = asyncio.run(run_async_code_eb83d9a2())
    logger.success(format_json(chat_history))
    return chat_history
chat_history = asyncio.run(run_async_code_657157e4())
logger.success(format_json(chat_history))

logger.debug(len(chat_history))

"""
Great, we can see that the current FIFO queue is only 2 messages (expected since we set the chat history token ratio to 0.02).

Now, let's inspect the long-term memory blocks that are inserted into the latest user message.
"""
logger.info("Great, we can see that the current FIFO queue is only 2 messages (expected since we set the chat history token ratio to 0.02).")

for block in chat_history[-2].blocks:
    logger.debug(block.text)

"""
To use this memory outside an agent, and to highlight more of the usage, you might do something like the following:
"""
logger.info("To use this memory outside an agent, and to highlight more of the usage, you might do something like the following:")

new_user_msg = ChatMessage(
    role="user", content="What kind of dog was I asking about?"
)
async def run_async_code_76c02065():
    await memory.aput(new_user_msg)
    return 
 = asyncio.run(run_async_code_76c02065())
logger.success(format_json())

async def run_async_code_99503262():
    async def run_async_code_bfdf8ff1():
        new_chat_history = await memory.aget()
        return new_chat_history
    new_chat_history = asyncio.run(run_async_code_bfdf8ff1())
    logger.success(format_json(new_chat_history))
    return new_chat_history
new_chat_history = asyncio.run(run_async_code_99503262())
logger.success(format_json(new_chat_history))
async def run_async_code_66a2865b():
    async def run_async_code_61da2084():
        resp = llm.chat(new_chat_history)
        return resp
    resp = asyncio.run(run_async_code_61da2084())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_66a2865b())
logger.success(format_json(resp))
async def run_async_code_8c2a4cc6():
    await memory.aput(resp.message)
    return 
 = asyncio.run(run_async_code_8c2a4cc6())
logger.success(format_json())
logger.debug(resp.message.content)

logger.info("\n\n[DONE]", bright=True)