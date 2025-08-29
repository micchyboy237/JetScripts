import asyncio
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from jet.llm.ollama.base import Ollama

initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/react_agent_with_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# ReAct Agent with Query Engine (RAG) Tools

In this section, we show how to setup an agent powered by the ReAct loop for financial analysis.

The agent has access to two "tools": one to query the 2021 Lyft 10-K and the other to query the 2021 Uber 10-K.

We try two different LLMs:

- gpt-3.5-turbo
- gpt-3.5-turbo-instruct

Note that you can plug in any LLM that exposes a text completion endpoint.
"""

"""
## Build Query Engine Tools
"""

# %pip install llama-index-llms-ollama


try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/lyft"
    )
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/uber"
    )
    uber_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

"""
Download Data
"""

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

if not index_loaded:
    lyft_docs = SimpleDirectoryReader(
        input_files=["./data/10k/lyft_2021.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=["./data/10k/uber_2021.pdf"]
    ).load_data()

    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    lyft_index.storage_context.persist(persist_dir="./storage/lyft")
    uber_index.storage_context.persist(persist_dir="./storage/uber")

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

"""
## Setup ReAct Agent

Here we setup two ReAct agents: one powered by standard gpt-3.5-turbo, and the other powered by gpt-3.5-turbo-instruct.

You can **optionally** specify context which will be added to the core ReAct system prompt.
"""


llm = Ollama(model="llama3.2")

agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
)

response = agent.chat("What was Lyft's revenue growth in 2021?")
print(str(response))

"""
## Run Some Example Queries

We run some example queries using the agent, showcasing some of the agent's abilities to do chain-of-thought-reasoning and tool use to synthesize the right answer.

We also show queries.
"""

response = agent.chat(
    "Compare and contrast the revenue growth of Uber and Lyft in 2021, then"
    " give an analysis"
)
print(str(response))

"""
**Async execution**: Here we try another query with async execution
"""

# import nest_asyncio

# nest_asyncio.apply()


async def async_func_4():
    response = agent.chat(
        "Compare and contrast the risks of Uber and Lyft in 2021, then give an"
        " analysis"
    )
    return response
response = asyncio.run(async_func_4())
logger.success(format_json(response))
print(str(response))

"""
### Compare gpt-3.5-turbo vs. gpt-3.5-turbo-instruct 

We compare the performance of the two agents in being able to answer some complex queries.
"""

"""
#### Taking a look at a turbo-instruct agent
"""

llm_instruct = Ollama(
    model="llama3.2")
agent_instruct = ReActAgent.from_tools(
    query_engine_tools, llm=llm_instruct, verbose=True
)

response = agent_instruct.chat("What was Lyft's revenue growth in 2021?")
print(str(response))

"""
#### Try more complex queries

We compare gpt-3.5-turbo with gpt-3.5-turbo-instruct agents on more complex queries.
"""

response = agent.chat(
    "Compare and contrast the revenue growth of Uber and Lyft in 2021, then"
    " give an analysis"
)
print(str(response))

response = agent_instruct.chat(
    "Compare and contrast the revenue growth of Uber and Lyft in 2021, then"
    " give an analysis"
)
print(str(response))

response = agent.chat(
    "Can you tell me about the risk factors of the company with the higher"
    " revenue?"
)
print(str(response))

response = agent_instruct.query(
    "Can you tell me about the risk factors of the company with the higher"
    " revenue?"
)
print(str(response))

"""
**Observation**: The turbo-instruct agent seems to do worse on agent reasoning compared to the regular turbo model. Of course, this is subject to further observation!
"""

logger.info("\n\n[DONE]", bright=True)
