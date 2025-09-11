from jet.logger import logger
from langchain_weaviate import WeaviateVectorStore
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Weaviate

>[Weaviate](https://weaviate.io/) is an open-source vector database. It allows you to store data objects and vector embeddings from
>your favorite ML models, and scale seamlessly into billions of data objects.


What is `Weaviate`?
- Weaviate is an open-source ​database of the type ​vector search engine.
- Weaviate allows you to store JSON documents in a class property-like fashion while attaching machine learning vectors to these documents to represent them in vector space.
- Weaviate can be used stand-alone (aka bring your vectors) or with a variety of modules that can do the vectorization for you and extend the core capabilities.
- Weaviate has a GraphQL-API to access your data easily.
- We aim to bring your vector search set up to production to query in mere milliseconds (check our [open-source benchmarks](https://weaviate.io/developers/weaviate/current/benchmarks/) to see if Weaviate fits your use case).
- Get to know Weaviate in the [basics getting started guide](https://weaviate.io/developers/weaviate/current/core-knowledge/basics.html) in under five minutes.

**Weaviate in detail:**

`Weaviate` is a low-latency vector search engine with out-of-the-box support for different media types (text, images, etc.). It offers Semantic Search, Question-Answer Extraction, Classification, Customizable Models (PyTorch/TensorFlow/Keras), etc. Built from scratch in Go, Weaviate stores both objects and vectors, allowing for combining vector search with structured filtering and the fault tolerance of a cloud-native database. It is all accessible through GraphQL, REST, and various client-side programming languages.

## Installation and Setup

Install the Python SDK:
"""
logger.info("# Weaviate")

pip install langchain-weaviate

"""
## Vector Store

There exists a wrapper around `Weaviate` indexes, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
"""
logger.info("## Vector Store")


"""
For a more detailed walkthrough of the Weaviate wrapper, see [this notebook](/docs/integrations/vectorstores/weaviate)
"""
logger.info("For a more detailed walkthrough of the Weaviate wrapper, see [this notebook](/docs/integrations/vectorstores/weaviate)")

logger.info("\n\n[DONE]", bright=True)