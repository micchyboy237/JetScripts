from llama_index.core.node_parser import TokenTextSplitter
from jet.llm.ollama.base import OllamaEmbedding
from pathway.xpacks.llm.vector_store import VectorStoreServer
import pathway as pw
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.pathway import PathwayRetriever
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/pathway_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pathway Retriever

# > [Pathway](https://pathway.com/) is an open data processing framework. It allows you to easily develop data transformation pipelines and Machine Learning applications that work with live data sources and changing data.
#
# This notebook demonstrates how to use a live data indexing pipeline with `LlamaIndex`. You can query the results of this pipeline from your LLM application using the provided `PathwayRetriever`. However, under the hood, Pathway updates the index on each data change giving you always up-to-date answers.
#
# In this notebook, we will use a [public demo document processing pipeline](https://pathway.com/solutions/ai-pipelines#try-it-out) that:
#
# 1. Monitors several cloud data sources for data changes.
# 2. Builds a vector index for the data.
#
# To have your own document processing pipeline check the [hosted offering](https://pathway.com/solutions/ai-pipelines) or [build your own](https://pathway.com/developers/user-guide/llm-xpack/vectorstore_pipeline/) by following this notebook.
#
# We will connect to the index using `llama_index.retrievers.pathway.PathwayRetriever` retriever, which implements the `retrieve` interface.
#
# The basic pipeline described in this document allows to effortlessly build a simple index of files stored in a cloud location. However, Pathway provides everything needed to build realtime data pipelines and apps, including SQL-like able operations such as groupby-reductions and joins between disparate data sources, time-based grouping and windowing of data, and a wide array of connectors.
#
# For more details about Pathway data ingestion pipeline and vector store, visit [vector store pipeline](https://pathway.com/developers/showcases/vectorstore_pipeline).

# Prerequisites
#
# To use `PathwayRetrievier` you must install `llama-index-retrievers-pathway` package.

# !pip install llama-index-retrievers-pathway

# Create Retriever for llama-index

# To instantiate and configure `PathwayRetriever` you need to provide either the `url` or the `host` and `port` of your document indexing pipeline. In the code below we use a publicly available [demo pipeline](https://pathway.com/solutions/ai-pipelines#try-it-out), which REST API you can access at `https://demo-document-indexing.pathway.stream`. This demo ingests documents from [Google Drive](https://drive.google.com/drive/u/0/folders/1cULDv2OaViJBmOfG5WB0oWcgayNrGtVs) and [Sharepoint](https://navalgo.sharepoint.com/sites/ConnectorSandbox/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FConnectorSandbox%2FShared%20Documents%2FIndexerSandbox&p=true&ga=1) and maintains an index for retrieving documents.


retriever = PathwayRetriever(
    url="https://demo-document-indexing.pathway.stream"
)
retriever.retrieve(str_or_query_bundle="what is pathway")

# **Your turn!** [Get your pipeline](https://pathway.com/solutions/ai-pipelines) or upload [new documents](https://chat-realtime-sharepoint-gdrive.demo.pathway.com/) to the demo pipeline and retry the query!

# Use in Query Engine


query_engine = RetrieverQueryEngine.from_args(
    retriever,
)

response = query_engine.query("Tell me about Pathway")
print(str(response))

# Building your own data processing pipeline

# Prerequisites
#
# Install `pathway` package. Then download sample data.

# %pip install pathway
# %pip install llama-index-embeddings-ollama

# !mkdir -p 'data/'
# !wget 'https://gist.githubusercontent.com/janchorowski/dd22a293f3d99d1b726eedc7d46d2fc0/raw/pathway_readme.md' -O 'data/pathway_readme.md'

# Define data sources tracked by Pathway

# Pathway can listen to many sources simultaneously, such as local files, S3 folders, cloud storage and any data stream for data changes.
#
# See [pathway-io](https://pathway.com/developers/api-docs/pathway-io) for more information.


data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./data",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )  # This creates a `pathway` connector that tracks
)

# Create the document indexing pipeline

# Let us create the document indexing pipeline. The `transformations` should be a list of `TransformComponent`s ending with an `Embedding` transformation.
#
# In this example, let's first split the text first using `TokenTextSplitter`, then embed with `OllamaEmbedding`.


embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

transformations_example = [
    TokenTextSplitter(
        chunk_size=150,
        chunk_overlap=10,
        separator=" ",
    ),
    embed_model,
]

processing_pipeline = VectorStoreServer.from_llamaindex_components(
    *data_sources,
    transformations=transformations_example,
)

PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8754

processing_pipeline.run_server(
    host=PATHWAY_HOST, port=PATHWAY_PORT, with_cache=False, threaded=True
)

# Connect the retriever to the custom pipeline


retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT)
retriever.retrieve(str_or_query_bundle="what is pathway")

logger.info("\n\n[DONE]", bright=True)
