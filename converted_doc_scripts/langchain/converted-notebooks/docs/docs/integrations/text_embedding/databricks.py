from jet.transformers.formatters import format_json
from databricks_langchain import DatabricksEmbeddings
from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
import asyncio
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
---
sidebar_label: Databricks
---

# DatabricksEmbeddings

> [Databricks](https://www.databricks.com/) Lakehouse Platform unifies data, analytics, and AI on one platform.

This notebook provides a quick overview for getting started with Databricks [embedding models](/docs/concepts/embedding_models). For detailed documentation of all `DatabricksEmbeddings` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.databricks.DatabricksEmbeddings.html).



## Overview
### Integration details

| Class | Package |
| :--- | :--- |
| [DatabricksEmbeddings](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.databricks.DatabricksEmbeddings.html) | [databricks-langchain](https://python.langchain.com/docs/integrations/providers/databricks/) |

### Supported Methods

`DatabricksEmbeddings` supports all methods of `Embeddings` class including async APIs.


### Endpoint Requirement

The serving endpoint `DatabricksEmbeddings` wraps must have Ollama-compatible embedding input/output format ([reference](https://mlflow.org/docs/latest/llms/deployments/index.html#embeddings)). As long as the input format is compatible, `DatabricksEmbeddings` can be used for any endpoint type hosted on [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html):

1. Foundation Models - Curated list of state-of-the-art foundation models such as BAAI General Embedding (BGE). These endpoint are ready to use in your Databricks workspace without any set up.
2. Custom Models - You can also deploy custom embedding models to a serving endpoint via MLflow with
your choice of framework such as LangChain, Pytorch, Transformers, etc.
3. External Models - Databricks endpoints can serve models that are hosted outside Databricks as a proxy, such as proprietary model service like Ollama text-embedding-3.


## Setup

To access Databricks models you'll need to create a Databricks account, set up credentials (only if you are outside Databricks workspace), and install required packages.

### Credentials (only if you are outside Databricks)

If you are running LangChain app inside Databricks, you can skip this step.

Otherwise, you need manually set the Databricks workspace hostname and personal access token to `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables, respectively. See [Authentication Documentation](https://docs.databricks.com/en/dev-tools/auth/index.html#databricks-personal-access-tokens) for how to get an access token.
"""
logger.info("# DatabricksEmbeddings")

# import getpass

os.environ["DATABRICKS_HOST"] = "https://your-workspace.cloud.databricks.com"
if "DATABRICKS_TOKEN" not in os.environ:
#     os.environ["DATABRICKS_TOKEN"] = getpass.getpass(
        "Enter your Databricks access token: "
    )

"""
### Installation

The LangChain Databricks integration lives in the `databricks-langchain` package:
"""
logger.info("### Installation")

# %pip install -qU databricks-langchain

"""
## Instantiation
"""
logger.info("## Instantiation")


embeddings = DatabricksEmbeddings(
    endpoint="databricks-bge-large-en",
)

"""
## Indexing and Retrieval

Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/rag).

Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.
"""
logger.info("## Indexing and Retrieval")


text = "LangChain is the framework for building context-aware reasoning applications"

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

retrieved_document = retriever.invoke("What is LangChain?")

retrieved_document[0].page_content

"""
## Direct Usage

Under the hood, the vectorstore and retriever implementations are calling `embeddings.embed_documents(...)` and `embeddings.embed_query(...)` to create embeddings for the text(s) used in `from_texts` and retrieval `invoke` operations, respectively.

You can directly call these methods to get embeddings for your own use cases.

### Embed single texts

You can embed single texts or documents with `embed_query`:
"""
logger.info("## Direct Usage")

single_vector = embeddings.embed_query(text)
logger.debug(str(single_vector)[:100])  # Show the first 100 characters of the vector

"""
### Embed multiple texts

You can embed multiple texts with `embed_documents`:
"""
logger.info("### Embed multiple texts")

text2 = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs"
)
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    logger.debug(str(vector)[:100])  # Show the first 100 characters of the vector

"""
### Async Usage

You can also use `aembed_query` and `aembed_documents` for producing embeddings asynchronously:
"""
logger.info("### Async Usage")



async def async_example():
    single_vector = await embeddings.aembed_query(text)
    logger.success(format_json(single_vector))
    logger.debug(str(single_vector)[:100])  # Show the first 100 characters of the vector


asyncio.run(async_example())

"""
## API Reference

For detailed documentation on `DatabricksEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.databricks.DatabricksEmbeddings.html).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)