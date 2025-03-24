"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/RouterQueryEngine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Router Query Engine
In this tutorial, we define a custom router query engine that selects one out of several candidate query engines to execute a query.
"""

"""
### Setup
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-embeddings-ollama
# %pip install llama-index-llms-ollama

# !pip install llama-index


from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex
from llama_index.core import SummaryIndex
from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from jet.llm.ollama.base import Ollama
import os
import nest_asyncio
nest_asyncio.apply()

"""
## Global Models
"""


# os.environ["OPENAI_API_KEY"] = "sk-..."


Settings.llm = Ollama(model="llama3.2", request_timeout=300.0,
                      context_window=4096, temperature=0.2)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

"""
### Load Data

We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.
"""


documents = SimpleDirectoryReader(
    "./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


Settings.chunk_size = 1024
nodes = Settings.node_parser.get_nodes_from_documents(documents)


storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

"""
### Define Summary Index and Vector Index over Same Data
"""


summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
### Define Query Engines and Set Metadata
"""

list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()


list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description=(
        "Useful for summarization questions related to Paul Graham eassy on"
        " What I Worked On."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)

"""
### Define Router Query Engine

There are several selectors available, each with some distinct attributes.

The LLM selectors use the LLM to output a JSON that is parsed, and the corresponding indexes are queried.

The Pydantic selectors (currently only supported by `gpt-4-0613` and `gpt-3.5-turbo-0613` (the default)) use the Ollama Function Call API to produce pydantic selection objects, rather than parsing raw JSON.

For each type of selector, there is also the option to select 1 index to route to, or multiple.

#### PydanticSingleSelector

Use the Ollama Function API to generate/parse pydantic objects under the hood for the router selector.
"""


query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")
print(str(response))

response = query_engine.query("What did Paul Graham do after RICS?")
print(str(response))

"""
#### LLMSingleSelector

Use Ollama(or any other LLM) to parse generated JSON under the hood to select a sub-index for routing.
"""

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")
print(str(response))

response = query_engine.query("What did Paul Graham do after RICS?")
print(str(response))

print(str(response.metadata["selector_result"]))

"""
#### PydanticMultiSelector

In case you are expecting queries to be routed to multiple indexes, you should use a multi selector. The multi selector sends to query to multiple sub-indexes, and then aggregates all responses using a summary index to form a complete answer.
"""


keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context using keywords from Paul"
        " Graham essay on What I Worked On."
    ),
)

query_engine = RouterQueryEngine(
    selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
        keyword_tool,
    ],
)

response = query_engine.query(
    "What were noteable events and people from the authors time at Interleaf"
    " and YC?"
)
print(str(response))

print(str(response.metadata["selector_result"]))
