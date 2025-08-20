import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import (
FunctionAgent,
ReActAgent,
)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/mistralai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MistralAI Cookbook

MistralAI released [mixtral-8x22b](https://mistral.ai/news/mixtral-8x22b/).

It is a sparse Mixture-of-Experts (SMoE) model that uses only 39B active parameters out of 141B, offering unparalleled cost efficiency for its size with 64K tokens context window, multilingual, strong maths coding, coding and Function calling capabilities.

This is a cook-book in showcasing the usage of `mixtral-8x22b` model with llama-index.

### Setup LLM and Embedding Model
"""
logger.info("# MistralAI Cookbook")

# import nest_asyncio

# nest_asyncio.apply()


os.environ["MISTRAL_API_KEY"] = "<YOUR MISTRAL API KEY>"


llm = MistralAI(model="open-mixtral-8x22b", temperature=0.1)
embed_model = MistralAIEmbedding(model_name="mistral-embed")

Settings.llm = llm
Settings.embed_model = embed_model

"""
### Download Data

We will use `Uber-2021` and `Lyft-2021` 10K SEC filings.
"""
logger.info("### Download Data")

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O './uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O './lyft_2021.pdf'

"""
### Load Data
"""
logger.info("### Load Data")


uber_docs = SimpleDirectoryReader(input_files=["./uber_2021.pdf"]).load_data()
lyft_docs = SimpleDirectoryReader(input_files=["./lyft_2021.pdf"]).load_data()

"""
### Build RAG on uber and lyft docs
"""
logger.info("### Build RAG on uber and lyft docs")


uber_index = VectorStoreIndex.from_documents(uber_docs)
uber_query_engine = uber_index.as_query_engine(similarity_top_k=5)

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
lyft_query_engine = lyft_index.as_query_engine(similarity_top_k=5)

response = uber_query_engine.query("What is the revenue of uber in 2021?")
logger.debug(response)

response = lyft_query_engine.query("What are lyft investments in 2021?")
logger.debug(response)

"""
### `FunctionAgent` with RAG QueryEngineTools.

Here we use `Fuction Calling` capabilities of the model.
"""
logger.info("### `FunctionAgent` with RAG QueryEngineTools.")


query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_query_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description="Provides information about Lyft financials for year 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=uber_query_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description="Provides information about Uber financials for year 2021",
        ),
    ),
]

agent = FunctionAgent(
    tools=query_engine_tools,
    llm=llm,
)

async def run_async_code_65ab4991():
    async def run_async_code_ee7aa79c():
        response = await agent.run("What is the revenue of uber in 2021.")
        return response
    response = asyncio.run(run_async_code_ee7aa79c())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_65ab4991())
logger.success(format_json(response))

logger.debug(str(response))

async def run_async_code_fc9cc280():
    async def run_async_code_4c877f08():
        response = await agent.run("What are lyft investments in 2021?")
        return response
    response = asyncio.run(run_async_code_4c877f08())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_fc9cc280())
logger.success(format_json(response))

logger.debug(str(response))

"""
### Agents and Tools usage
"""
logger.info("### Agents and Tools usage")


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)

"""
### With Function Calling.
"""
logger.info("### With Function Calling.")

agent = FunctionAgent(
    tools=[multiply_tool, add_tool, subtract_tool],
    llm=llm,
)

async def run_async_code_f4945e7d():
    async def run_async_code_15efd760():
        response = await agent.run("What is (26 * 2) + 2024?")
        return response
    response = asyncio.run(run_async_code_15efd760())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_f4945e7d())
logger.success(format_json(response))
logger.debug(str(response))

"""
### With ReAct Agent
"""
logger.info("### With ReAct Agent")

agent = ReActAgent(tools=[multiply_tool, add_tool, subtract_tool], llm=llm)

async def run_async_code_f4945e7d():
    async def run_async_code_15efd760():
        response = await agent.run("What is (26 * 2) + 2024?")
        return response
    response = asyncio.run(run_async_code_15efd760())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_f4945e7d())
logger.success(format_json(response))
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)