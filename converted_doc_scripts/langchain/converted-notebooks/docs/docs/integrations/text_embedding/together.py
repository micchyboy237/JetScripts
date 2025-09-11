from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_together import TogetherEmbeddings
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
sidebar_label: Together AI
---

# TogetherEmbeddings

This will help you get started with Together embedding models using LangChain. For detailed documentation on `TogetherEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/together/embeddings/langchain_together.embeddings.TogetherEmbeddings.html).

## Overview
### Integration details


<ItemTable category="text_embedding" item="Together" />

## Setup

To access Together embedding models you'll need to create a/an Together account, get an API key, and install the `langchain-together` integration package.

### Credentials

Head to [https://api.together.xyz/](https://api.together.xyz/) to sign up to Together and generate an API key. Once you've done this set the TOGETHER_API_KEY environment variable:
"""
logger.info("# TogetherEmbeddings")

# import getpass

if not os.getenv("TOGETHER_API_KEY"):
#     os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter your Together API key: ")

"""
To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:")



"""
### Installation

The LangChain Together integration lives in the `langchain-together` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-together

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
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

retrieved_documents = retriever.invoke("What is LangChain?")

retrieved_documents[0].page_content

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
## API Reference

For detailed documentation on `TogetherEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/together/embeddings/langchain_together.embeddings.TogetherEmbeddings.html).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)