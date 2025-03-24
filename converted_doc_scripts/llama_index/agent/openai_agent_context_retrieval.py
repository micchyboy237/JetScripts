from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
import json
from typing import Sequence
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Document
from llama_index.agent.openai_legacy import ContextRetrieverOpenAIAgent
from llama_index.core.tools import BaseTool, FunctionTool

initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_context_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Context-Augmented Ollama Agent

In this tutorial, we show you how to use our `ContextRetrieverOpenAIAgent` implementation
to build an agent on top of Ollama's function API and store/index an arbitrary number of tools. Our indexing/retrieval modules help to remove the complexity of having too many functions to fit in the prompt.

## Initial Setup

Here we setup a ContextRetrieverOpenAIAgent. This agent will perform retrieval first before calling any tools. This can help ground the agent's tool picking and answering capabilities in context.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-agent-openai-legacy

# !pip install llama-index


try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/march"
    )
    march_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/june"
    )
    june_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/sept"
    )
    sept_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

"""
Download Data
"""

# !mkdir -p 'data/10q/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_march_2022.pdf' -O 'data/10q/uber_10q_march_2022.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_june_2022.pdf' -O 'data/10q/uber_10q_june_2022.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_sept_2022.pdf' -O 'data/10q/uber_10q_sept_2022.pdf'

if not index_loaded:
    march_docs = SimpleDirectoryReader(
        input_files=["./data/10q/uber_10q_march_2022.pdf"]
    ).load_data()
    june_docs = SimpleDirectoryReader(
        input_files=["./data/10q/uber_10q_june_2022.pdf"]
    ).load_data()
    sept_docs = SimpleDirectoryReader(
        input_files=["./data/10q/uber_10q_sept_2022.pdf"]
    ).load_data()

    march_index = VectorStoreIndex.from_documents(march_docs)
    june_index = VectorStoreIndex.from_documents(june_docs)
    sept_index = VectorStoreIndex.from_documents(sept_docs)

    march_index.storage_context.persist(persist_dir="./storage/march")
    june_index.storage_context.persist(persist_dir="./storage/june")
    sept_index.storage_context.persist(persist_dir="./storage/sept")

march_engine = march_index.as_query_engine(similarity_top_k=3)
june_engine = june_index.as_query_engine(similarity_top_k=3)
sept_engine = sept_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=march_engine,
        metadata=ToolMetadata(
            name="uber_march_10q",
            description=(
                "Provides information about Uber 10Q filings for March 2022. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=june_engine,
        metadata=ToolMetadata(
            name="uber_june_10q",
            description=(
                "Provides information about Uber financials for June 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=sept_engine,
        metadata=ToolMetadata(
            name="uber_sept_10q",
            description=(
                "Provides information about Uber financials for Sept 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

"""
### Try Context-Augmented Agent

Here we augment our agent with context in different settings:
- toy context: we define some abbreviations that map to financial terms (e.g. R=Revenue). We supply this as context to the agent
"""


texts = [
    "Abbreviation: X = Revenue",
    "Abbreviation: YZ = Risk Factors",
    "Abbreviation: Z = Costs",
]
docs = [Document(text=t) for t in texts]
context_index = VectorStoreIndex.from_documents(docs)

context_agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
    query_engine_tools,
    context_index.as_retriever(similarity_top_k=1),
    verbose=True,
)

response = context_agent.chat("What is the YZ of March 2022?")

logger.debug(str(response))

context_agent.chat("What is the X and Z in September 2022?")

"""
### Use Uber 10-Q as context, use Calculator as Tool
"""


def magic_formula(revenue: int, cost: int) -> int:
    """Runs MAGIC_FORMULA on revenue and cost."""
    return revenue - cost


magic_tool = FunctionTool.from_defaults(fn=magic_formula, name="magic_formula")

context_agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
    [magic_tool], sept_index.as_retriever(similarity_top_k=3), verbose=True
)

response = context_agent.chat(
    "Can you run MAGIC_FORMULA on Uber's revenue and cost?"
)

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
