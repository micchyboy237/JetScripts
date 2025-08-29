from jet.llm.utils.llama_index_utils import display_jet_source_node
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.tools import RetrieverTool
from jet.llm.ollama.base import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SummaryIndex
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    SimpleKeywordTableIndex,
)
import sys
import logging
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/router_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Router Retriever
# In this guide, we define a custom router retriever that selects one or more candidate retrievers in order to execute a given query.
#
# The router (`BaseSelector`) module uses the LLM to dynamically make decisions on which underlying retrieval tools to use. This can be helpful to select one out of a diverse range of data sources. This can also be helpful to aggregate retrieval results across a variety of data sources (if a multi-selector module is used).
#
# This notebook is very similar to the RouterQueryEngine notebook.

# Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-llms-ollama

# !pip install llama-index


nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Download Data

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# Load Data
#
# We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()

llm = Ollama(model="llama3.1")
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

list_retriever = summary_index.as_retriever()
vector_retriever = vector_index.as_retriever()
keyword_retriever = keyword_index.as_retriever()


list_tool = RetrieverTool.from_defaults(
    retriever=list_retriever,
    description=(
        "Will retrieve all context from Paul Graham's essay on What I Worked"
        " On. Don't use if the question only requires more specific context."
    ),
)
vector_tool = RetrieverTool.from_defaults(
    retriever=vector_retriever,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)
keyword_tool = RetrieverTool.from_defaults(
    retriever=keyword_retriever,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On (using entities mentioned in query)"
    ),
)

# Define Selector Module for Routing
#
# There are several selectors available, each with some distinct attributes.
#
# The LLM selectors use the LLM to output a JSON that is parsed, and the corresponding indexes are queried.
#
# The Pydantic selectors (currently only supported by `gpt-4-0613` and `gpt-3.5-turbo-0613` (the default)) use the Ollama Function Call API to produce pydantic selection objects, rather than parsing raw JSON.
#
# Here we use PydanticSingleSelector/PydanticMultiSelector but you can use the LLM-equivalents as well.


# PydanticSingleSelector

retriever = RouterRetriever(
    selector=PydanticSingleSelector.from_defaults(llm=llm),
    retriever_tools=[
        list_tool,
        vector_tool,
    ],
)

nodes = retriever.retrieve(
    "Can you give me all the context regarding the author's life?"
)
for node in nodes:
    display_jet_source_node(node)

nodes = retriever.retrieve("What did Paul Graham do after RISD?")
for node in nodes:
    display_jet_source_node(node)

# PydanticMultiSelector

retriever = RouterRetriever(
    selector=PydanticMultiSelector.from_defaults(llm=llm),
    retriever_tools=[list_tool, vector_tool, keyword_tool],
)

nodes = retriever.retrieve(
    "What were noteable events from the authors time at Interleaf and YC?"
)
for node in nodes:
    display_jet_source_node(node)

nodes = retriever.retrieve(
    "What were noteable events from the authors time at Interleaf and YC?"
)
for node in nodes:
    display_jet_source_node(node)

nodes = await retriever.aretrieve(
    "What were noteable events from the authors time at Interleaf and YC?"
)
for node in nodes:
    display_jet_source_node(node)

logger.info("\n\n[DONE]", bright=True)
