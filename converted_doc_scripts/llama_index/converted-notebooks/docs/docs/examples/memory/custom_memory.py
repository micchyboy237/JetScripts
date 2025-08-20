import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.llms import LLM, ChatMessage, TextBlock, ImageBlock
from llama_index.core.memory import Memory
from llama_index.core.memory import Memory, StaticMemoryBlock
from llama_index.core.workflow import (
Context,
Event,
StartEvent,
StopEvent,
Workflow,
step,
)
from pydantic import Field
from typing import List, Literal, Optional
import os
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Manipulating Memory at Runtime

In this notebook, we cover how to use the `Memory` class to build an agentic workflow with dynamic memory.

Specifically, we will build a workflow where a user can upload a file, and pin that to the context of the LLM (i.e. like the file context in Cursor).

By default, as the short-term memory fills up and is flushed, it will be passed to memory blocks for processing as needed (extracting facts, indexing for retrieval, or for static blocks, ignoring it).

With this notebook, the intent is to show how memory can be managed and manipulated at runtime, beyond the already existing functionality described above.

## Setup

For our workflow, we will use MLX as our LLM.
"""
logger.info("# Manipulating Memory at Runtime")

# !pip install llama-index-core llama-index-llms-ollama


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
## Workflow Setup

Our workflow will be fairly straightfoward. There will be two main entry points

1. Adding/Removing files from memory 
2. Chatting with the LLM

Using the `Memory` class, we can introduce memory blocks that hold our static context.
"""
logger.info("## Workflow Setup")



class InitEvent(StartEvent):
    user_msg: str
    new_file_paths: List[str] = Field(default_factory=list)
    removed_file_paths: List[str] = Field(default_factory=list)


class ContextUpdateEvent(Event):
    new_file_paths: List[str] = Field(default_factory=list)
    removed_file_paths: List[str] = Field(default_factory=list)


class ChatEvent(Event):
    pass


class ResponseEvent(StopEvent):
    response: str


class ContextualLLMChat(Workflow):
    def __init__(self, memory: Memory, llm: LLM, **workflow_kwargs):
        super().__init__(**workflow_kwargs)
        self._memory = memory
        self._llm = llm

    def _path_to_block_name(self, file_path: str) -> str:
        return re.sub(r"[^\w-]", "_", file_path)

    @step
    async def init(self, ev: InitEvent) -> ContextUpdateEvent | ChatEvent:
        async def run_async_code_5ec470f3():
            await self._memory.aput(ChatMessage(role="user", content=ev.user_msg))
            return 
         = asyncio.run(run_async_code_5ec470f3())
        logger.success(format_json())

        if ev.new_file_paths or ev.removed_file_paths:
            return ContextUpdateEvent(
                new_file_paths=ev.new_file_paths,
                removed_file_paths=ev.removed_file_paths,
            )
        else:
            return ChatEvent()

    @step
    async def update_memory_context(self, ev: ContextUpdateEvent) -> ChatEvent:
        current_blocks = self._memory.memory_blocks
        current_block_names = [block.name for block in current_blocks]

        for new_file_path in ev.new_file_paths:
            if new_file_path not in current_block_names:
                if new_file_path.endswith((".png", ".jpg", ".jpeg")):
                    self._memory.memory_blocks.append(
                        StaticMemoryBlock(
                            name=self._path_to_block_name(new_file_path),
                            static_content=[ImageBlock(path=new_file_path)],
                        )
                    )
                elif new_file_path.endswith((".txt", ".md", ".py", ".ipynb")):
                    with open(new_file_path, "r") as f:
                        self._memory.memory_blocks.append(
                            StaticMemoryBlock(
                                name=self._path_to_block_name(new_file_path),
                                static_content=f.read(),
                            )
                        )
                else:
                    raise ValueError(f"Unsupported file: {new_file_path}")
        for removed_file_path in ev.removed_file_paths:
            named_block = self._path_to_block_name(removed_file_path)
            self._memory.memory_blocks = [
                block
                for block in self._memory.memory_blocks
                if block.name != named_block
            ]

        return ChatEvent()

    @step
    async def chat(self, ev: ChatEvent) -> ResponseEvent:
        async def run_async_code_630e56f2():
            async def run_async_code_09dfa0eb():
                chat_history = await self._memory.aget()
                return chat_history
            chat_history = asyncio.run(run_async_code_09dfa0eb())
            logger.success(format_json(chat_history))
            return chat_history
        chat_history = asyncio.run(run_async_code_630e56f2())
        logger.success(format_json(chat_history))
        async def run_async_code_22f7d43f():
            async def run_async_code_167fe807():
                response = self._llm.chat(chat_history)
                return response
            response = asyncio.run(run_async_code_167fe807())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_22f7d43f())
        logger.success(format_json(response))
        return ResponseEvent(response=response.message.content)

"""
## Using the Workflow

Now that we have our chat workflow defined, we can try it out! You can use any file, but for this example, we will use a few dummy files.
"""
logger.info("## Using the Workflow")

# !wget https://mediaproxy.tvtropes.org/width/1200/https://static.tvtropes.org/pmwiki/pub/images/shrek_cover.png -O ./image.png
# !wget https://raw.githubusercontent.com/run-llama/llama_index/refs/heads/main/llama-index-core/llama_index/core/memory/memory.py -O ./memory.py


llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

memory = Memory.from_defaults(
    session_id="my_session",
    token_limit=60000,
    chat_history_token_ratio=0.7,
    token_flush_size=5000,
    insert_method="user",
)

workflow = ContextualLLMChat(
    memory=memory,
    llm=llm,
    verbose=True,
)

"""
We can simulate a user adding a file to memory, and then chatting with the LLM.
"""
logger.info("We can simulate a user adding a file to memory, and then chatting with the LLM.")

async def async_func_0():
    response = await workflow.run(
        user_msg="What does this file contain?",
        new_file_paths=["./memory.py"],
    )
    return response
response = asyncio.run(async_func_0())
logger.success(format_json(response))

logger.debug("--------------------------------")
logger.debug(response.response)

"""
Great! Now, we can simulate a user removing that file, and adding a new one.
"""
logger.info("Great! Now, we can simulate a user removing that file, and adding a new one.")

async def async_func_0():
    response = await workflow.run(
        user_msg="What does this next file contain?",
        new_file_paths=["./image.png"],
        removed_file_paths=["./memory.py"],
    )
    return response
response = asyncio.run(async_func_0())
logger.success(format_json(response))

logger.debug("--------------------------------")
logger.debug(response.response)

"""
It works! Now, you've learned how to manage memory in a custom workflow. Beyond just letting short-term memory flush into memory blocks, you can manually manipulate the memory blocks at runtime as well.
"""
logger.info("It works! Now, you've learned how to manage memory in a custom workflow. Beyond just letting short-term memory flush into memory blocks, you can manually manipulate the memory blocks at runtime as well.")

logger.info("\n\n[DONE]", bright=True)