from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_netmind import NetmindEmbeddings
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
sidebar_label: Netmind
---

# NetmindEmbeddings

This will help you get started with Netmind embedding models using LangChain. For detailed documentation on `NetmindEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/).

## Overview
### Integration details

| Provider | Package |
|:--------:|:-------:|
| [Netmind](/docs/integrations/providers/netmind/) | [langchain-netmind](https://python.langchain.com/api_reference/) |

## Setup

To access Netmind embedding models you'll need to create a/an Netmind account, get an API key, and install the `langchain-netmind` integration package.

### Credentials

Head to https://www.netmind.ai/ to sign up to Netmind and generate an API key. Once you've done this set the NETMIND_API_KEY environment variable:
"""
logger.info("# NetmindEmbeddings")

# import getpass

if not os.getenv("NETMIND_API_KEY"):
#     os.environ["NETMIND_API_KEY"] = getpass.getpass("Enter your Netmind API key: ")

"""
If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

The LangChain Netmind integration lives in the `langchain-netmind` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-netmind

"""
## Instantiation

Now we can instantiate our model object:
"""
logger.info("## Instantiation")


embeddings = NetmindEmbeddings(
    model="nvidia/NV-Embed-v2",
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

For detailed documentation on `NetmindEmbeddings` features and configuration options, please refer to the:  
* [API reference](https://python.langchain.com/api_reference/)  
* [langchain-netmind](https://github.com/protagolabs/langchain-netmind)  
* [pypi](https://pypi.org/project/langchain-netmind/)
"""
logger.info("## API Reference")


logger.info("\n\n[DONE]", bright=True)