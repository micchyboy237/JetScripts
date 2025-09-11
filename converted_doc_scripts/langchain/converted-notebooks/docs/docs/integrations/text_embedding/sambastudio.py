from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_sambanova import SambaStudioEmbeddings
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
sidebar_label: SambaStudio
---

# SambaStudioEmbeddings

This will help you get started with SambaNova's SambaStudio embedding models using LangChain. For detailed documentation on `SambaStudioEmbeddings` features and configuration options, please refer to the [API reference](https://docs.sambanova.ai/sambastudio/latest/index.html).

**[SambaNova](https://sambanova.ai/)'s** [SambaStudio](https://sambanova.ai/technology/full-stack-ai-platform) is a platform for running your own open-source models

## Overview
### Integration details

| Provider | Package |
|:--------:|:-------:|
| [SambaNova](/docs/integrations/providers/sambanova/) | [langchain-sambanova](https://python.langchain.com/docs/integrations/providers/sambanova/) |

## Setup

To access SambaStudio models you will need to [deploy an endpoint](https://docs.sambanova.ai/sambastudio/latest/language-models.html) in your SambaStudio platform, install the `langchain_sambanova` integration package.

```bash
pip install langchain-sambanova
```

### Credentials

Get the URL and API Key from your SambaStudio deployed endpoint and add them to your environment variables:

``` bash
export SAMBASTUDIO_URL="sambastudio-url-key-here"
export SAMBASTUDIO_API_KEY="your-api-key-here"
```
"""
logger.info("# SambaStudioEmbeddings")

# import getpass

if not os.getenv("SAMBASTUDIO_URL"):
#     os.environ["SAMBASTUDIO_URL"] = getpass.getpass(
        "Enter your SambaStudio endpoint URL: "
    )

if not os.getenv("SAMBASTUDIO_API_KEY"):
#     os.environ["SAMBASTUDIO_API_KEY"] = getpass.getpass(
        "Enter your SambaStudio API key: "
    )

"""
If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

The LangChain SambaNova integration lives in the `langchain-sambanova` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-sambanova

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


embeddings = SambaStudioEmbeddings(
    model="e5-mistral-7b-instruct",
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

For detailed documentation on `SambaStudio` features and configuration options, please refer to the [API reference](https://docs.sambanova.ai/sambastudio/latest/api-ref-landing.html).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)