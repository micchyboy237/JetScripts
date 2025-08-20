from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SummaryIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.pathway import PathwayReader
from pathway.xpacks.llm.vector_store import VectorStoreServer
import logging
import os
import pathway as pw
import shutil
import sys


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/PathwayReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pathway Reader

> [Pathway](https://pathway.com/) is an open data processing framework. It allows you to easily develop data transformation pipelines and Machine Learning applications that work with live data sources and changing data.

This notebook demonstrates how to set up a live data indexing pipeline. You can query the results of this pipeline from your LLM application in the same manner as you would a regular reader. However, under the hood, Pathway updates the index on each data change giving you always up-to-date answers.

In this notebook, we will first connect the `llama_index.readers.pathway.PathwayReader` reader to a [public demo document processing pipeline](https://pathway.com/solutions/ai-pipelines#try-it-out) that:

1. Monitors several cloud data sources for data changes.
2. Builds a vector index for the data.

To have your own document processing pipeline check the [hosted offering](https://pathway.com/solutions/ai-pipelines) or [build your own](https://pathway.com/developers/user-guide/llm-xpack/vectorstore_pipeline/) by following this notebook. 

The basic pipeline described in this document allows to effortlessly build a simple index of files stored in a cloud location. However, Pathway provides everything needed to build realtime data pipelines and apps, including SQL-like able operations such as groupby-reductions and joins between disparate data sources, time-based grouping and windowing of data, and a wide array of connectors. 

For more details about Pathway data ingestion pipeline and vector store, visit [vector store pipeline](https://pathway.com/developers/user-guide/llm-xpack/vectorstore_pipeline/).

## Prerequisites

Install the `llama-index-readers-pathway` integration
"""
logger.info("# Pathway Reader")

# %pip install llama-index-readers-pathway

"""
Configure logging
"""
logger.info("Configure logging")


logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
Set up your MLX API key.
"""
logger.info("Set up your MLX API key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("MLX API Key:")

"""
## Create the reader and connect to a public pipeline

To instantiate and configure `PathwayReader` you need to provide either the `url` or the `host` and `port` of your document indexing pipeline. In the code below we use a publicly available [demo pipeline](https://pathway.com/solutions/ai-pipelines#try-it-out), which REST API you can access at `https://demo-document-indexing.pathway.stream`. This demo ingests documents from [Google Drive](https://drive.google.com/drive/u/0/folders/1cULDv2OaViJBmOfG5WB0oWcgayNrGtVs) and [Sharepoint](https://navalgo.sharepoint.com/sites/ConnectorSandbox/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FConnectorSandbox%2FShared%20Documents%2FIndexerSandbox&p=true&ga=1) and maintains an index for retrieving documents.
"""
logger.info("## Create the reader and connect to a public pipeline")


reader = PathwayReader(url="https://demo-document-indexing.pathway.stream")
reader.load_data(query_text="What is Pathway")

"""
## Create a summary index with llama-index
"""
logger.info("## Create a summary index with llama-index")

docs = reader.load_data(query_text="What is Pathway", k=2)

index = SummaryIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("What does Pathway do?")
logger.debug(response)

"""
## Building your own data processing pipeline

### Prerequisites

Install `pathway` package. Then download sample data.
"""
logger.info("## Building your own data processing pipeline")

# %pip install pathway
# %pip install llama-index-embeddings-ollama

# !mkdir -p 'data/'
# !wget 'https://gist.githubusercontent.com/janchorowski/dd22a293f3d99d1b726eedc7d46d2fc0/raw/pathway_readme.md' -O 'data/pathway_readme.md'

"""
### Define data sources tracked by Pathway

Pathway can listen to many sources simultaneously, such as local files, S3 folders, cloud storage and any data stream for data changes.

See [pathway-io](https://pathway.com/developers/api-docs/pathway-io) for more information.
"""
logger.info("### Define data sources tracked by Pathway")


data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./data",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )  # This creates a `pathway` connector that tracks
)

"""
### Create the document indexing pipeline

Let us create the document indexing pipeline. The `transformations` should be a list of `TransformComponent`s ending with an `Embedding` transformation.

In this example, let's first split the text first using `TokenTextSplitter`, then embed with `MLXEmbedding`.
"""
logger.info("### Create the document indexing pipeline")


embed_model = MLXEmbedding(embed_batch_size=10)

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

"""
### Connect the reader to the custom pipeline
"""
logger.info("### Connect the reader to the custom pipeline")


reader = PathwayReader(host=PATHWAY_HOST, port=PATHWAY_PORT)
reader.load_data(query_text="What is Pathway")

logger.info("\n\n[DONE]", bright=True)