from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
import os
import shutil
import {ItemTable} from "@theme/FeatureTables"


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
sidebar_label: Ollama
---

# OllamaEmbeddings

This will help you get started with Ollama embedding models using LangChain. For detailed documentation on `OllamaEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/ollama/embeddings/jet.adapters.langchain.chat_ollama.embeddings.OllamaEmbeddings.html).

## Overview
### Integration details


<ItemTable category="text_embedding" item="Ollama" />

## Setup

First, follow [these instructions](https://github.com/ollama/ollama?tab=readme-ov-file#ollama) to set up and run a local Ollama instance:

* [Download](https://ollama.ai/download) and install Ollama onto the available supported platforms (including Windows Subsystem for Linux aka WSL, macOS, and Linux)
    * macOS users can install via Homebrew with `brew install ollama` and start with `brew services start ollama`
* Fetch available LLM model via `ollama pull <name-of-model>`
    * View a list of available models via the [model library](https://ollama.ai/library)
    * e.g., `ollama pull llama3`
* This will download the default tagged version of the model. Typically, the default points to the latest, smallest sized-parameter model.

> On Mac, the models will be download to `~/.ollama/models`
>
> On Linux (or WSL), the models will be stored at `/usr/share/ollama/.ollama/models`

* Specify the exact version of the model of interest as such `ollama pull vicuna:13b-v1.5-16k-q4_0` (View the [various tags for the `Vicuna`](https://ollama.ai/library/vicuna/tags) model in this instance)
* To view all pulled models, use `ollama list`
* To chat directly with a model from the command line, use `ollama run <name-of-model>`
* View the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs) for more commands. You can run `ollama help` in the terminal to see available commands.

To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("# OllamaEmbeddings")


"""
### Installation

The LangChain Ollama integration lives in the `langchain-ollama` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-ollama

"""
## Instantiation

Now we can instantiate our model object and generate embeddings:
"""
logger.info("## Instantiation")


embeddings = OllamaEmbeddings(
    model="llama3",
)

"""
## Indexing and Retrieval

Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/rag/).

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

logger.debug(retrieved_documents[0].page_content)

"""
## Direct Usage

Under the hood, the vectorstore and retriever implementations are calling `embeddings.embed_documents(...)` and `embeddings.embed_query(...)` to create embeddings for the text(s) used in `from_texts` and retrieval `invoke` operations, respectively.

You can directly call these methods to get embeddings for your own use cases.

### Embed single texts

You can embed single texts or documents with `embed_query`:
"""
logger.info("## Direct Usage")

single_vector = embeddings.embed_query(text)
# Show the first 100 characters of the vector
logger.debug(str(single_vector)[:100])

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
    # Show the first 100 characters of the vector
    logger.debug(str(vector)[:100])

"""
## API Reference

For detailed documentation on `OllamaEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/ollama/embeddings/jet.adapters.langchain.chat_ollama.embeddings.OllamaEmbeddings.html).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)
