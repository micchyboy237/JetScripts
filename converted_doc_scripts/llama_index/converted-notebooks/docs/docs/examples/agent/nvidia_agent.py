import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.agent.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
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
# Function Calling NVIDIA Agent

This notebook shows you how to use our NVIDIA agent, powered by function calling capabilities.

## Initial Setup

Let's start by importing some simple building blocks.  

The main thing we need is:
1. the NVIDIA NIM Endpoint (using our own `llama_index` LLM class)
2. a place to keep conversation history 
3. a definition for tools that our agent can use.
"""
logger.info("# Function Calling NVIDIA Agent")

# %pip install --upgrade --quiet llama-index-llms-nvidia

# import getpass

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    logger.debug("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
#     nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key


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
Here we initialize a simple NVIDIA agent with calculator functions.
"""
logger.info("Here we initialize a simple NVIDIA agent with calculator functions.")

llm = NVIDIA("meta/llama-3.1-70b-instruct")


agent = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
)

"""
### Chat
"""
logger.info("### Chat")

async def run_async_code_efa6c44f():
    async def run_async_code_cddfcfeb():
        response = await agent.run("What is (121 * 3) + 42?")
        return response
    response = asyncio.run(run_async_code_cddfcfeb())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_efa6c44f())
logger.success(format_json(response))
logger.debug(str(response))

logger.debug(response.tool_calls)

"""
### Managing Context/Memory

By default, `.run()` is stateless. If you want to maintain state, you can pass in a `context` object.
"""
logger.info("### Managing Context/Memory")


ctx = Context(agent)

async def run_async_code_dbae8d91():
    async def run_async_code_b09b8983():
        response = await agent.run("Hello, my name is John Doe.", ctx=ctx)
        return response
    response = asyncio.run(run_async_code_b09b8983())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_dbae8d91())
logger.success(format_json(response))
logger.debug(str(response))

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
### Agent with Personality

You can specify a system prompt to give the agent additional instruction or personality.
"""
logger.info("### Agent with Personality")

agent = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt="Talk like a pirate in every response.",
)

async def run_async_code_62af0b70():
    async def run_async_code_6e013bea():
        response = await agent.run("Hi")
        return response
    response = asyncio.run(run_async_code_6e013bea())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_62af0b70())
logger.success(format_json(response))
logger.debug(response)

async def run_async_code_f9238355():
    async def run_async_code_79a35496():
        response = await agent.run("Tell me a story")
        return response
    response = asyncio.run(run_async_code_79a35496())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_f9238355())
logger.success(format_json(response))
logger.debug(response)

"""
# NVIDIA Agent with RAG/Query Engine Tools
"""
logger.info("# NVIDIA Agent with RAG/Query Engine Tools")

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'


embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

uber_index = VectorStoreIndex.from_documents(
    uber_docs, embed_model=embed_model
)
uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=llm)
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=uber_engine,
    name="uber_10k",
    description=(
        "Provides information about Uber financials for year 2021. "
        "Use a detailed plain text question as input to the tool."
    ),
)

agent = FunctionAgent(tools=[query_engine_tool], llm=llm)

async def async_func_27():
    response = await agent.run(
        "Tell me both the risk factors and tailwinds for Uber? Do two parallel tool calls."
    )
    return response
response = asyncio.run(async_func_27())
logger.success(format_json(response))
logger.debug(str(response))

"""
# ReAct Agent
"""
logger.info("# ReAct Agent")


agent = ReActAgent([multiply_tool, add_tool], llm=llm, verbose=True)

"""
Using the `stream_events()` method, we can stream the response as it is generated to see the agent's thought process.

The final response will have only the final answer.
"""
logger.info("Using the `stream_events()` method, we can stream the response as it is generated to see the agent's thought process.")


handler = agent.run("What is 20+(2*4)? Calculate step by step ")
async for ev in handler.stream_events():
    if isinstance(ev, AgentStream):
        logger.debug(ev.delta, end="", flush=True)

async def run_async_code_78ff5c2b():
    async def run_async_code_2cbcd794():
        response = await handler
        return response
    response = asyncio.run(run_async_code_2cbcd794())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_78ff5c2b())
logger.success(format_json(response))

logger.debug(str(response))

logger.debug(response.tool_calls)

logger.info("\n\n[DONE]", bright=True)