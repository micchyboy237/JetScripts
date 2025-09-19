from jet.models.config import MODELS_CACHE_DIR
from jet.transformers.formatters import format_json
from IPython.display import display, Markdown
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.embeddings.azure_openai import AzureHuggingFaceEmbedding
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.azure_openai import AzureOllamaFunctionCallingAdapter
import logging
import os
import pprint
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/knowledge_graph_rag_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Knowledge Graph RAG Query Engine


## Graph RAG

Graph RAG is an Knowledge-enabled RAG approach to retrieve information from Knowledge Graph on given task. Typically, this is to build context based on entities' SubGraph related to the task.

## GraphStore backed RAG vs VectorStore RAG

As we compared how Graph RAG helps in some use cases in [this tutorial](https://gpt-index.readthedocs.io/en/latest/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html#id1), it's shown Knowledge Graph as the unique format of information could mitigate several issues caused by the nature of the "split and embedding" RAG approach.

## Why Knowledge Graph RAG Query Engine

In Llama Index, there are two scenarios we could apply Graph RAG:

- Build Knowledge Graph from documents with Llama Index, with LLM or even [local models](https://colab.research.google.com/drive/1G6pcR0pXvSkdMQlAK_P-IrYgo-_staxd?usp=sharing), to do this, we should go for `KnowledgeGraphIndex`.
- Leveraging existing Knowledge Graph, in this case, we should use `KnowledgeGraphRAGQueryEngine`.

> Note, the third query engine that's related to KG in Llama Index is `NL2GraphQuery` or `Text2Cypher`, for either exiting KG or not, it could be done with `KnowledgeGraphQueryEngine`.

Before we start the `Knowledge Graph RAG QueryEngine` demo, let's first get ready for basic preparation of Llama Index.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Knowledge Graph RAG Query Engine")

# %pip install llama-index-llms-azure-openai
# %pip install llama-index-graph-stores-nebula
# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-azure-openai

# !pip install llama-index

"""
### OllamaFunctionCalling
"""
logger.info("### OllamaFunctionCalling")


# os.environ["OPENAI_API_KEY"] = "sk-..."


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output


Settings.llm = OllamaFunctionCalling(temperature=0, model="llama3.2")
Settings.chunk_size = 512

"""
### Azure
"""
logger.info("### Azure")


api_key = "<api-key>"
azure_endpoint = "https://<your-resource-name>.openai.azure.com/"
api_version = "2023-07-01-preview"

llm = AzureOllamaFunctionCallingAdapter(
    model="gpt-35-turbo-16k",
    deployment_name="my-custom-llm",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureHuggingFaceEmbedding(
    model="text-embedding-ada-002",
    deployment_name="my-custom-embedding",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)


Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

"""
## Prepare for NebulaGraph

We take [NebulaGraphStore](https://gpt-index.readthedocs.io/en/stable/examples/index_structs/knowledge_graph/NebulaGraphKGIndexDemo.html) as an example in this demo, thus before next step to perform Graph RAG on existing KG, let's ensure we have a running NebulaGraph with defined data schema.

This step installs the clients of NebulaGraph, and prepare contexts that defines a [NebulaGraph Graph Space](https://docs.nebula-graph.io/3.6.0/1.introduction/2.data-model/).
"""
logger.info("## Prepare for NebulaGraph")

# %pip install ipython-ngql nebula3-python

os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"  # default is "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"  # assumed we have NebulaGraph installed locally

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

"""
Then we could instiatate a `NebulaGraphStore`, in order to create a `StorageContext`'s `graph_store` as it.
"""
logger.info("Then we could instiatate a `NebulaGraphStore`, in order to create a `StorageContext`'s `graph_store` as it.")


graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

"""
Here, we assumed to have the same Knowledge Graph from [this tutorial](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/knowledge_graph_query_engine.html#optional-build-the-knowledge-graph-with-llamaindex)

## Perform Graph RAG Query

Finally, let's demo how to do Graph RAG towards an existing Knowledge Graph.

All we need to do is to use `RetrieverQueryEngine` and configure the retriver of it to be `KnowledgeGraphRAGRetriever`.

The `KnowledgeGraphRAGRetriever` performs the following steps:

- Search related Entities of the quesion/task
- Get SubGraph of those Entities (default 2-depth) from the KG
- Build Context based on the SubGraph

Please note, the way to Search related Entities could be either Keyword extraction based or Embedding based, which is controlled by argument `retriever_mode` of the `KnowledgeGraphRAGRetriever`, and supported options are:
- "keyword"
- "embedding"(not yet implemented)
- "keyword_embedding"(not yet implemented)

Here is the example on how to use `RetrieverQueryEngine` and `KnowledgeGraphRAGRetriever`:
"""
logger.info("## Perform Graph RAG Query")


graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever,
)

"""
Then we can query it like:
"""
logger.info("Then we can query it like:")


response = query_engine.query(
    "Tell me about Peter Quill?",
)
display(Markdown(f"<b>{response}</b>"))

response = query_engine.query(
    "Tell me about Peter Quill?",
)
logger.success(format_json(response))
display(Markdown(f"<b>{response}</b>"))

"""
## Include nl2graphquery as Context in Graph RAG

The nature of (Sub)Graph RAG and nl2graphquery are different. No one is better than the other but just when one fits more in certain type of questions. To understand more on how they differ from the other, see [this demo](https://www.siwei.io/en/demos/graph-rag/) comparing the two.

<video width="938" height="800" 
       src="https://github.com/siwei-io/talks/assets/1651790/05d01e53-d819-4f43-9bf1-75549f7f2be9"  
       controls>
</video>

While in real world cases, we may not always know which approach works better, thus, one way to best leverage KG in RAG are fetching both retrieval results as context and letting LLM + Prompt generate answer with them all being involved.

So, optionally, we could choose to synthesise answer from two piece of retrieved context from KG:
- Graph RAG, the default retrieval method, which extracts subgraph that's related to the key entities in the question.
- NL2GraphQuery, generate Knowledge Graph Query based on query and the Schema of the Knowledge Graph, which is by default switched off.

We could set `with_nl2graphquery=True` to enable it like:
"""
logger.info("## Include nl2graphquery as Context in Graph RAG")

graph_rag_retriever_with_nl2graphquery = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
    with_nl2graphquery=True,
)

query_engine_with_nl2graphquery = RetrieverQueryEngine.from_args(
    graph_rag_retriever_with_nl2graphquery,
)

response = query_engine_with_nl2graphquery.query(
    "What do you know about Peter Quill?",
)
display(Markdown(f"<b>{response}</b>"))

"""
And let's check the response's metadata to know more details of the retrival of Graph RAG with nl2graphquery by inspecting `response.metadata`.

- **text2Cypher**, it generates a Cypher Query towards the answer as the context.

```cypher
Graph Store Query: MATCH (e:`entity`)-[r:`relationship`]->(e2:`entity`)
WHERE e.`entity`.`name` == 'Peter Quill'
RETURN e2.`entity`.`name`
```
- **SubGraph RAG**, it get the SubGraph of 'Peter Quill' to build the context.

- Finally, it combined the two nodes of context, to synthesize the answer.
"""
logger.info("And let's check the response's metadata to know more details of the retrival of Graph RAG with nl2graphquery by inspecting `response.metadata`.")


pp = pprint.PrettyPrinter()
pp.plogger.debug(response.metadata)

logger.info("\n\n[DONE]", bright=True)
