from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_lindorm_integration import LindormAIEmbeddings
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
sidebar_label: Lindorm
---

# LindormAIEmbeddings

This will help you get started with Lindorm embedding models using LangChain. 

## Overview
### Integration details

| Provider |              Package              |
|:--------:|:---------------------------------:|
| [Lindorm](/docs/integrations/providers/lindorm/) | [langchain-lindorm-integration](https://pypi.org/project/langchain-lindorm-integration/) |

## Setup


To access Lindorm embedding models you'll need to create a Lindorm account, get AK&SK, and install the `langchain-lindorm-integration` integration package.

### Credentials


You can get you credentials in the [console](https://lindorm.console.aliyun.com/cn-hangzhou/clusterhou/cluster?spm=a2c4g.11186623.0.0.466534e93Xj6tt)
"""
logger.info("# LindormAIEmbeddings")



class Config:
    AI_LLM_ENDPOINT = os.environ.get("AI_ENDPOINT", "<AI_ENDPOINT>")
    AI_USERNAME = os.environ.get("AI_USERNAME", "root")
    AI_PWD = os.environ.get("AI_PASSWORD", "<PASSWORD>")

    AI_DEFAULT_EMBEDDING_MODEL = "bge_m3_model"  # set to your deployed model

"""
### Installation

The LangChain Lindorm integration lives in the `langchain-lindorm-integration` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-lindorm-integration

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


embeddings = LindormAIEmbeddings(
    endpoint=Config.AI_LLM_ENDPOINT,
    username=Config.AI_USERNAME,
    password=Config.AI_PWD,
    model_name=Config.AI_DEFAULT_EMBEDDING_MODEL,
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

For detailed documentation on `LindormEmbeddings` features and configuration options, please refer to the [API reference](https://pypi.org/project/langchain-lindorm-integration/).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)