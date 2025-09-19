from jet.models.config import MODELS_CACHE_DIR
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleKeywordTableIndex
from llama_index.core import StorageContext
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import (
PydanticMultiSelector,
PydanticSingleSelector,
)
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/RouterQueryEngine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Router Query Engine
In this tutorial, we define a custom router query engine that selects one out of several candidate query engines to execute a query.

### Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Router Query Engine")

# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-ollama

# !pip install llama-index

# import nest_asyncio

# nest_asyncio.apply()

"""
## Global Models
"""
logger.info("## Global Models")


# os.environ["OPENAI_API_KEY"] = "sk-..."


Settings.llm = OllamaFunctionCalling(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.2)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

"""
### Load Data

We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.
"""
logger.info("### Load Data")


documents = SimpleDirectoryReader("./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


Settings.chunk_size = 1024
nodes = Settings.node_parser.get_nodes_from_documents(documents)


storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

"""
### Define Summary Index and Vector Index over Same Data
"""
logger.info("### Define Summary Index and Vector Index over Same Data")


summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
### Define Query Engines and Set Metadata
"""
logger.info("### Define Query Engines and Set Metadata")

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

The Pydantic selectors (currently only supported by `gpt-4-0613` and `gpt-3.5-turbo-0613` (the default)) use the OllamaFunctionCalling Function Call API to produce pydantic selection objects, rather than parsing raw JSON.

For each type of selector, there is also the option to select 1 index to route to, or multiple.

#### PydanticSingleSelector

Use the OllamaFunctionCalling Function API to generate/parse pydantic objects under the hood for the router selector.
"""
logger.info("### Define Router Query Engine")



query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")
logger.debug(str(response))

response = query_engine.query("What did Paul Graham do after RICS?")
logger.debug(str(response))

"""
#### LLMSingleSelector

Use OllamaFunctionCalling (or any other LLM) to parse generated JSON under the hood to select a sub-index for routing.
"""
logger.info("#### LLMSingleSelector")

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")
logger.debug(str(response))

response = query_engine.query("What did Paul Graham do after RICS?")
logger.debug(str(response))

logger.debug(str(response.metadata["selector_result"]))

"""
#### PydanticMultiSelector

In case you are expecting queries to be routed to multiple indexes, you should use a multi selector. The multi selector sends to query to multiple sub-indexes, and then aggregates all responses using a summary index to form a complete answer.
"""
logger.info("#### PydanticMultiSelector")


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
logger.debug(str(response))

logger.debug(str(response.metadata["selector_result"]))

logger.info("\n\n[DONE]", bright=True)