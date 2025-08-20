import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import (
ToolCall,
ToolCallResult,
AgentStream,
)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os
import shutil


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/ollama.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Ollama + `gpt-oss` Cookbook

MLX's latest open-source models, `gpt-oss`, [have been released](https://openai.com/open-models/).

They come in two sizes:
- 20 billion parameter model
- 120 billion parameter model

These models are Apache 2.0 licensed, and can be run locally on your machine. In this cookbook, we will use Ollama to demonstrate capabilities and test some claims of agentic and chain-of-thought behavior.

## Setup
First, follow the [readme](https://github.com/jmorganca/ollama) to set up and run a local Ollama instance.

When the Ollama app is running on your local machine:
- All of your local models are automatically served on localhost:11434
- Select your model when setting llm = Ollama(..., model="<model family>:<version>")
- Increase defaullt timeout (30 seconds) if needed setting Ollama(..., request_timeout=300.0)
- If you set llm = Ollama(..., model="<model family") without a version it will simply look for latest
- By default, the maximum context window for your model is used. You can manually set the `context_window` to limit memory usage.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Ollama + `gpt-oss` Cookbook")

# %pip install llama-index-llms-ollama

"""
## Chain-of-thought / Thinking with `gpt-oss`

Ollama supports configuration for thinking when using `gpt-oss` models. Let's test this out with a few examples.
"""
logger.info("## Chain-of-thought / Thinking with `gpt-oss`")


llm = Ollama(
    model="gpt-oss:20b",
    request_timeout=360,
    thinking=True,
    temperature=1.0,
    context_window=8000,
)

async def run_async_code_16c794ea():
    async def run_async_code_b9ed4eb3():
        resp_gen = llm.stream_complete("What is 1234 * 5678?")
        return resp_gen
    resp_gen = asyncio.run(run_async_code_b9ed4eb3())
    logger.success(format_json(resp_gen))
    return resp_gen
resp_gen = asyncio.run(run_async_code_16c794ea())
logger.success(format_json(resp_gen))

still_thinking = True
logger.debug("====== THINKING ======")
async for chunk in resp_gen:
    if still_thinking and chunk.additional_kwargs.get("thinking_delta"):
        logger.debug(chunk.additional_kwargs["thinking_delta"], end="", flush=True)
    elif still_thinking:
        still_thinking = False
        logger.debug("\n====== ANSWER ======")

    if not still_thinking:
        logger.debug(chunk.delta, end="", flush=True)

"""
## Creating agents with `gpt-oss`

While giving a response from a prompt is fine, we can also incorporate tools to get more precise results, and build an agent.
"""
logger.info("## Creating agents with `gpt-oss`")



def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


llm = Ollama(
    model="gpt-oss:20b",
    request_timeout=360,
    thinking=False,
    temperature=1.0,
    context_window=8000,
)

agent = FunctionAgent(
    tools=[multiply],
    llm=llm,
    system_prompt="You are a helpful assistant that can multiply and add numbers. Always rely on tools for math operations.",
)


handler = agent.run("What is 1234 * 5678?")
async for ev in handler.stream_events():
    if isinstance(ev, ToolCall):
        logger.debug(f"\nTool call: {ev.tool_name}({ev.tool_kwargs}")
    elif isinstance(ev, ToolCallResult):
        logger.debug(
            f"\nTool call: {ev.tool_name}({ev.tool_kwargs}) -> {ev.tool_output}"
        )
    elif isinstance(ev, AgentStream):
        logger.debug(ev.delta, end="", flush=True)

async def run_async_code_95d041c2():
    resp = await handler
    return resp
resp = asyncio.run(run_async_code_95d041c2())
logger.success(format_json(resp))

"""
### Remembering past events with Agents

By default, agent runs do not remember past events. However, using the `Context`, we can maintain state between calls.
"""
logger.info("### Remembering past events with Agents")


ctx = Context(agent)

async def run_async_code_de842f21():
    async def run_async_code_b1f6a9a8():
        resp = await agent.run("What is 1234 * 5678?", ctx=ctx)
        return resp
    resp = asyncio.run(run_async_code_b1f6a9a8())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_de842f21())
logger.success(format_json(resp))
async def run_async_code_1b178d5b():
    async def run_async_code_1c70ba8d():
        resp = await agent.run("What was the last question/answer pair?", ctx=ctx)
        return resp
    resp = asyncio.run(run_async_code_1c70ba8d())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_1b178d5b())
logger.success(format_json(resp))

logger.debug(resp.response.content)

logger.info("\n\n[DONE]", bright=True)