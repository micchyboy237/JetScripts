import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.memory import Memory, BaseMemoryBlock
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import Field
from typing import List, Optional, Any
import os
import shutil
import tiktoken


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Reducing Multi-Turn Confusion with LlamaIndex Memory

[Recent research](https://arxiv.org/abs/2505.06120) has shown the performance of an LLM significantly degrades given multi-turn conversations.

To help avoid this, we can implement a custom short-term and long-term memory in LlamaIndex to ensure that the conversation turns never get too long, and condense the memory as we go.

Using the code from this notebook, you may see improvements in your own agents as it works to limit how many turns are in your chat history.

**NOTE:** This notebook was tested with `llama-index-core>=0.12.37`, as that version included some fixes to make this work nicely.
"""
logger.info("# Reducing Multi-Turn Confusion with LlamaIndex Memory")

# %pip install -U llama-index-core llama-index-llms-ollama


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
## Setup

To make this work, we need two things
1. A memory block that condenses a;; past chat messages into a single string while maintaining a token limit
2. A `Memory` instance that uses that memory block, and has token limits configured such that multi-turn conversations are always flushed to the memory block for handling

First, the custom memory block:
"""
logger.info("## Setup")



class CondensedMemoryBlock(BaseMemoryBlock[str]):
    current_memory: List[str] = Field(default_factory=list)
    token_limit: int = Field(default=50000)
    tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(
        "qwen3-1.7b-4bit"
    )  # all openai models use 4o tokenizer these days

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        """Return the current memory block contents."""
        return "\n".join(self.current_memory)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Push messages into the memory block. (Only handles text content)"""
        for message in messages:
            text_contents = "\n".join(
                block.text
                for block in message.blocks
                if isinstance(block, TextBlock)
            )
            memory_str = f"<message role={message.role}>"

            if text_contents:
                memory_str += f"\n{text_contents}"

            kwargs = {
                key: val
                for key, val in message.additional_kwargs.items()
                if key != "session_id"
            }
            if kwargs:
                memory_str += f"\n({kwargs})"

            memory_str += "\n</message>"
            self.current_memory.append(memory_str)

        message_length = sum(
            len(self.tokenizer.encode(message))
            for message in self.current_memory
        )
        while message_length > self.token_limit:
            self.current_memory = self.current_memory[1:]
            message_length = sum(
                len(self.tokenizer.encode(message))
                for message in self.current_memory
            )

"""
And then, a `Memory` instance that uses that block while configuring a very limited token limit for the short-term memory:
"""
logger.info("And then, a `Memory` instance that uses that block while configuring a very limited token limit for the short-term memory:")

block = CondensedMemoryBlock(name="condensed_memory")

memory = Memory.from_defaults(
    session_id="test-mem-01",
    token_limit=60000,
    token_flush_size=5000,
    async_database_uri="sqlite+aiosqlite:///:memory:",
    memory_blocks=[block],
    insert_method="user",
    chat_history_token_ratio=0.0001,
)

"""
## Usage

Let's explore using this with some dummy messages, and observe how the memory is managed.
"""
logger.info("## Usage")

initial_messages = [
    ChatMessage(role="user", content="Hello! My name is Logan"),
    ChatMessage(role="assistant", content="Hello! How can I help you?"),
    ChatMessage(role="user", content="What is the capital of France?"),
    ChatMessage(role="assistant", content="The capital of France is Paris"),
]

async def run_async_code_da35f982():
    await memory.aput_messages(initial_messages)
    return 
 = asyncio.run(run_async_code_da35f982())
logger.success(format_json())

"""
Then, lets add our next user message!
"""
logger.info("Then, lets add our next user message!")

await memory.aput_messages(
    [ChatMessage(role="user", content="What was my name again?")]
)

"""
With that, we can explore what the chat history looks like before sending to an LLM.
"""
logger.info("With that, we can explore what the chat history looks like before sending to an LLM.")

async def run_async_code_657157e4():
    async def run_async_code_eb83d9a2():
        chat_history = await memory.aget()
        return chat_history
    chat_history = asyncio.run(run_async_code_eb83d9a2())
    logger.success(format_json(chat_history))
    return chat_history
chat_history = asyncio.run(run_async_code_657157e4())
logger.success(format_json(chat_history))

for message in chat_history:
    logger.debug(message.role)
    logger.debug(message.content)
    logger.debug()

"""
Great! Even though we added many messages, it gets condensed into a single user message!

Let's try with an actual agent next.

## Agent Usage

Here, we can create a `FunctionAgent` with some simple tools that uses our memory.
"""
logger.info("## Agent Usage")



def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

agent = FunctionAgent(
    tools=[multiply, divide, add, subtract],
    llm=llm,
    system_prompt="You are a helpful assistant that can do simple math operations with tools.",
)

block = CondensedMemoryBlock(name="condensed_memory")

memory = Memory.from_defaults(
    session_id="test-mem-01",
    token_limit=60000,
    token_flush_size=5000,
    async_database_uri="sqlite+aiosqlite:///:memory:",
    memory_blocks=[block],
    insert_method="user",
    chat_history_token_ratio=0.0001,
)

async def run_async_code_0d6178c5():
    async def run_async_code_2004ccae():
        resp = await agent.run("What is (3214 * 322) / 2?", memory=memory)
        return resp
    resp = asyncio.run(run_async_code_2004ccae())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_0d6178c5())
logger.success(format_json(resp))
logger.debug(resp)

async def run_async_code_b0fb598e():
    async def run_async_code_b24d0f84():
        current_chat_history = await memory.aget()
        return current_chat_history
    current_chat_history = asyncio.run(run_async_code_b24d0f84())
    logger.success(format_json(current_chat_history))
    return current_chat_history
current_chat_history = asyncio.run(run_async_code_b0fb598e())
logger.success(format_json(current_chat_history))
for message in current_chat_history:
    logger.debug(message.role)
    logger.debug(message.content)
    logger.debug()

"""
Perfect! Since the memory didn't have a new user message yet, it added one with our current memory. On the next user message, that memory and the user message would get combined like we saw earlier.

Let's try a few follow ups to confirm this is working properly
"""
logger.info("Perfect! Since the memory didn't have a new user message yet, it added one with our current memory. On the next user message, that memory and the user message would get combined like we saw earlier.")

async def async_func_0():
    resp = await agent.run(
        "What was the last question I asked you?", memory=memory
    )
    return resp
resp = asyncio.run(async_func_0())
logger.success(format_json(resp))
logger.debug(resp)

async def async_func_5():
    resp = await agent.run(
        "And how did you go about answering that message?", memory=memory
    )
    return resp
resp = asyncio.run(async_func_5())
logger.success(format_json(resp))
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)