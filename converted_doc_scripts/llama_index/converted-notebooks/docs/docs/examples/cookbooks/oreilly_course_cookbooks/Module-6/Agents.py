import asyncio
from jet.transformers.formatters import format_json
from IPython.display import display, HTML
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import (
FunctionAgent,
ReActAgent,
)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agents

### Installation
"""
logger.info("# Agents")

# !pip install llama-index

"""
### Setup LLM and Embedding Model
"""
logger.info("### Setup LLM and Embedding Model")

# import nest_asyncio

# nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "sk-..."


llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.1)
embed_model = MLXEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model

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
### With ReAct Agent
"""
logger.info("### With ReAct Agent")

agent = ReActAgent(
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

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

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

display(HTML(f'<p style="font-size:20px">{response}</p>'))

"""
## Agent with RAG Query Engine Tools

### Download Data

We will use `Uber-2021` and `Lyft-2021` 10K SEC filings.
"""
logger.info("## Agent with RAG Query Engine Tools")

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
uber_query_engine = uber_index.as_query_engine(similarity_top_k=3)

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
lyft_query_engine = lyft_index.as_query_engine(similarity_top_k=3)

response = uber_query_engine.query("What are the investments of Uber in 2021?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

response = lyft_query_engine.query("What are lyft investments in 2021?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

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

async def run_async_code_d277bf9b():
    async def run_async_code_3fad670a():
        response = await agent.run("What are the investments of Uber in 2021?")
        return response
    response = asyncio.run(run_async_code_3fad670a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_d277bf9b())
logger.success(format_json(response))

display(HTML(f'<p style="font-size:20px">{response}</p>'))

async def run_async_code_fc9cc280():
    async def run_async_code_4c877f08():
        response = await agent.run("What are lyft investments in 2021?")
        return response
    response = asyncio.run(run_async_code_4c877f08())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_fc9cc280())
logger.success(format_json(response))

display(HTML(f'<p style="font-size:20px">{response}</p>'))

logger.info("\n\n[DONE]", bright=True)