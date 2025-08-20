import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import ToolCallResult
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context
from llama_index.llms.anthropic import Anthropic
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/mistral_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Function Calling Anthropic Agent

This notebook shows you how to use our Anthropic agent, powered by function calling capabilities.

**NOTE:** Only claude-3* models support function calling using Anthropic's API.

## Initial Setup

Let's start by importing some simple building blocks.  

The main thing we need is:
1. the Anthropic API (using our own `llama_index` LLM class)
2. a place to keep conversation history 
3. a definition for tools that our agent can use.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Function Calling Anthropic Agent")

# %pip install llama-index
# %pip install llama-index-llms-anthropic
# %pip install llama-index-embeddings-ollama

"""
Let's define some very simple calculator tools for our agent.
"""
logger.info("Let's define some very simple calculator tools for our agent.")

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

"""
# Make sure your ANTHROPIC_API_KEY is set. Otherwise explicitly specify the `api_key` parameter.
"""
# logger.info("Make sure your ANTHROPIC_API_KEY is set. Otherwise explicitly specify the `api_key` parameter.")


llm = Anthropic(model="claude-3-opus-20240229", api_key="sk-...")

"""
## Initialize Anthropic Agent

Here we initialize a simple Anthropic agent with calculator functions.
"""
logger.info("## Initialize Anthropic Agent")


agent = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
)



async def run_agent_verbose(query: str):
    handler = agent.run(query)
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            logger.debug(
                f"Called tool {event.tool_name} with args {event.tool_kwargs}\nGot result: {event.tool_output}"
            )

    async def run_async_code_745df5ad():
        return await handler
        return 
     = asyncio.run(run_async_code_745df5ad())
    logger.success(format_json())

"""
### Chat
"""
logger.info("### Chat")

async def run_async_code_109fd87f():
    async def run_async_code_465d16ff():
        response = await run_agent_verbose("What is (121 + 2) * 5?")
        return response
    response = asyncio.run(run_async_code_465d16ff())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_109fd87f())
logger.success(format_json(response))
logger.debug(str(response))

logger.debug(response.tool_calls)

"""
### Managing Context/Memory

By default, `.run()` is stateless. If you want to maintain state, you can pass in a `context` object.
"""
logger.info("### Managing Context/Memory")


ctx = Context(agent)

async def run_async_code_80376ede():
    async def run_async_code_cde01853():
        response = await agent.run("My name is John Doe", ctx=ctx)
        return response
    response = asyncio.run(run_async_code_cde01853())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_80376ede())
logger.success(format_json(response))
async def run_async_code_43e0ff52():
    async def run_async_code_457d45ea():
        response = await agent.run("What is my name?", ctx=ctx)
        return response
    response = asyncio.run(run_async_code_457d45ea())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_43e0ff52())
logger.success(format_json(response))

logger.debug(str(response))

"""
## Anthropic Agent over RAG Pipeline

Build a Anthropic agent over a simple 10K document. We use MLX embeddings and claude-3-haiku-20240307 to construct the RAG pipeline, and pass it to the Anthropic Opus agent as a tool.
"""
logger.info("## Anthropic Agent over RAG Pipeline")

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'


embed_model = MLXEmbedding(
    model_name="text-embedding-3-large", api_key="sk-proj-..."
)
query_llm = Anthropic(model="claude-3-haiku-20240307", api_key="sk-...")

uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

uber_index = VectorStoreIndex.from_documents(
    uber_docs, embed_model=embed_model
)
uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=query_llm)
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=uber_engine,
    name="uber_10k",
    description=(
        "Provides information about Uber financials for year 2021. "
        "Use a detailed plain text question as input to the tool."
    ),
)


agent = FunctionAgent(tools=[query_engine_tool], llm=llm, verbose=True)

async def async_func_34():
    response = await agent.run(
        "Tell me both the risk factors and tailwinds for Uber?"
    )
    return response
response = asyncio.run(async_func_34())
logger.success(format_json(response))
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)