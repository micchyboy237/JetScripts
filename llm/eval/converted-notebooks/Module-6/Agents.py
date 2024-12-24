from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Agents
#

### Installation

# !pip install llama-index

### Setup LLM and Embedding Model

import nest_asyncio

nest_asyncio.apply()

import os

# os.environ["OPENAI_API_KEY"] = "sk-..."

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096, temperature=0.1)
embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

Settings.llm = llm
Settings.embed_model = embed_model

### Agents and Tools usage

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import (
    FunctionCallingAgentWorker,
    ReActAgent,
)

from IPython.display import display, HTML

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

### With ReAct Agent

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool], llm=llm, verbose=True
)

response = agent.chat("What is (26 * 2) + 2024?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

### With Function Calling.

agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, add_tool, subtract_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = agent_worker.as_agent()

response = agent.chat("What is (26 * 2) + 2024?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

## Agent with RAG Query Engine Tools

### Download Data
# 
# We will use `Uber-2021` and `Lyft-2021` 10K SEC filings.

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O './uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O './lyft_2021.pdf'

### Load Data

from llama_index.core import SimpleDirectoryReader

uber_docs = SimpleDirectoryReader(input_files=["./uber_2021.pdf"]).load_data()
lyft_docs = SimpleDirectoryReader(input_files=["./lyft_2021.pdf"]).load_data()

### Build RAG on uber and lyft docs

from llama_index.core import VectorStoreIndex

uber_index = VectorStoreIndex.from_documents(uber_docs)
uber_query_engine = uber_index.as_query_engine(similarity_top_k=3)

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
lyft_query_engine = lyft_index.as_query_engine(similarity_top_k=3)

response = uber_query_engine.query("What are the investments of Uber in 2021?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

response = lyft_query_engine.query("What are lyft investments in 2021?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

### `FunctionCallingAgent` with RAG QueryEngineTools.
# 
# Here we use `Fuction Calling` capabilities of the model.

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker

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

agent_worker = FunctionCallingAgentWorker.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = agent_worker.as_agent()

response = agent.chat("What are the investments of Uber in 2021?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

response = agent.chat("What are lyft investments in 2021?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

logger.info("\n\n[DONE]", bright=True)