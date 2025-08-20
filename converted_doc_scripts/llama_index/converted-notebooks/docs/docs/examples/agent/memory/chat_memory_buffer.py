import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Chat Memory Buffer

**NOTE:** This example of memory is deprecated in favor of the newer and more flexible `Memory` class. See the [latest docs](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/).

The `ChatMemoryBuffer` is a memory buffer that simply stores the last X messages that fit into a token limit.

%pip install llama-index-core

## Setup
"""
logger.info("# Chat Memory Buffer")


memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

"""
## Using Standalone
"""
logger.info("## Using Standalone")


chat_history = [
    ChatMessage(role="user", content="Hello, how are you?"),
    ChatMessage(role="assistant", content="I'm doing well, thank you!"),
]

memory.put_messages(chat_history)

history = memory.get()

all_history = memory.get_all()

memory.reset()

"""
## Using with Agents

You can set the memory in any agent in the `.run()` method.
"""
logger.info("## Using with Agents")


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."



memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

agent = FunctionAgent(tools=[], llm=MLX(model="qwen3-1.7b-4bit-mini"))

ctx = Context(agent)

async def run_async_code_94bdf377():
    async def run_async_code_bf67faa6():
        resp = await agent.run("Hello, how are you?", ctx=ctx, memory=memory)
        return resp
    resp = asyncio.run(run_async_code_bf67faa6())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_94bdf377())
logger.success(format_json(resp))

logger.debug(memory.get_all())

logger.info("\n\n[DONE]", bright=True)