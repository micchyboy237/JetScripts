from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_modelscope import ModelScopeEmbeddings
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
sidebar_label: ModelScope
---

# ModelScopeEmbeddings

ModelScope ([Home](https://www.modelscope.cn/) | [GitHub](https://github.com/modelscope/modelscope)) is built upon the notion of “Model-as-a-Service” (MaaS). It seeks to bring together most advanced machine learning models from the AI community, and streamlines the process of leveraging AI models in real-world applications. The core ModelScope library open-sourced in this repository provides the interfaces and implementations that allow developers to perform model inference, training and evaluation. 

This will help you get started with ModelScope embedding models using LangChain.

## Overview
### Integration details

| Provider | Package |
|:--------:|:-------:|
| [ModelScope](/docs/integrations/providers/modelscope/) | [langchain-modelscope-integration](https://pypi.org/project/langchain-modelscope-integration/) |

## Setup

To access ModelScope embedding models you'll need to create a/an ModelScope account, get an API key, and install the `langchain-modelscope-integration` integration package.

### Credentials

Head to [ModelScope](https://modelscope.cn/) to sign up to ModelScope.
"""
logger.info("# ModelScopeEmbeddings")

# import getpass

if not os.getenv("MODELSCOPE_SDK_TOKEN"):
#     os.environ["MODELSCOPE_SDK_TOKEN"] = getpass.getpass(
        "Enter your ModelScope SDK token: "
    )

"""
### Installation

The LangChain ModelScope integration lives in the `langchain-modelscope-integration` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-modelscope-integration

"""
## Instantiation

Now we can instantiate our model object:
"""
logger.info("## Instantiation")


embeddings = ModelScopeEmbeddings(
    model_id="damo/nlp_corom_sentence-embedding_english-base",
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

For detailed documentation on `ModelScopeEmbeddings` features and configuration options, please refer to the [API reference](https://www.modelscope.cn/docs/sdk/pipelines).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)