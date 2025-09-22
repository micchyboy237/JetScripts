from jet.logger import logger
from langchain_aimlapi import AimlapiEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import os
import shutil
import { ItemTable } from "@theme/FeatureTables";


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
sidebar_label: AI/ML API Embeddings
---

# AimlapiEmbeddings

This will help you get started with AI/ML API embedding models using LangChain. For detailed documentation on `AimlapiEmbeddings` features and configuration options, please refer to the [API reference](https://docs.aimlapi.com/?utm_source=langchain&utm_medium=github&utm_campaign=integration).

## Overview
### Integration details


<ItemTable category="text_embedding" item="AI/ML API" />

## Setup

To access AI/ML API embedding models you'll need to create an account, get an API key, and install the `langchain-aimlapi` integration package.

### Credentials

Head to [https://aimlapi.com/app/](https://aimlapi.com/app/?utm_source=langchain&utm_medium=github&utm_campaign=integration) to sign up and generate an API key. Once you've done this, set the `AIMLAPI_API_KEY` environment variable:
"""
logger.info("# AimlapiEmbeddings")

# import getpass

if not os.getenv("AIMLAPI_API_KEY"):
#     os.environ["AIMLAPI_API_KEY"] = getpass.getpass("Enter your AI/ML API key: ")

"""
To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:")



"""
### Installation

The LangChain AI/ML API integration lives in the `langchain-aimlapi` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-aimlapi

"""
## Instantiation

Now we can instantiate our embeddings model and perform embedding operations:
"""
logger.info("## Instantiation")


embeddings = AimlapiEmbeddings(
    model="text-embedding-ada-002",
)

"""
## Indexing and Retrieval

Embedding models are often used in retrieval-augmented generation (RAG) flows. Below is how to index and retrieve data using the `embeddings` object we initialized above with `InMemoryVectorStore`.
"""
logger.info("## Indexing and Retrieval")


text = "LangChain is the framework for building context-aware reasoning applications"

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

retrieved_documents = retriever.invoke("What is LangChain?")
retrieved_documents[0].page_content

"""
## Direct Usage

You can directly call `embed_query` and `embed_documents` for custom embedding scenarios.

### Embed single text:
"""
logger.info("## Direct Usage")

single_vector = embeddings.embed_query(text)
logger.debug(str(single_vector)[:100])

"""
### Embed multiple texts:
"""
logger.info("### Embed multiple texts:")

text2 = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs"
)
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    logger.debug(str(vector)[:100])

"""
## API Reference

For detailed documentation on `AimlapiEmbeddings` features and configuration options, please refer to the [API reference](https://docs.aimlapi.com/?utm_source=langchain&utm_medium=github&utm_campaign=integration).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)