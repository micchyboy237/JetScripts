from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_naver import ClovaXEmbeddings
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
sidebar_label: Naver
---

# ClovaXEmbeddings

This notebook covers how to get started with embedding models provided by CLOVA Studio. For detailed documentation on `ClovaXEmbeddings` features and configuration options, please refer to the [API reference](https://guide.ncloud-docs.com/docs/clovastudio-dev-langchain#%EC%9E%84%EB%B2%A0%EB%94%A9%EB%8F%84%EA%B5%AC%EC%9D%B4%EC%9A%A9).

## Overview
### Integration details

| Provider | Package |
|:--------:|:-------:|
| [Naver](/docs/integrations/providers/naver.mdx) | [langchain-naver](https://pypi.org/project/langchain-naver/) |

## Setup

Before using embedding models provided by CLOVA Studio, you must go through the three steps below.

1. Creating [NAVER Cloud Platform](https://www.ncloud.com/) account 
2. Apply to use [CLOVA Studio](https://www.ncloud.com/product/aiService/clovaStudio)
3. Create a CLOVA Studio Test App or Service App of a model to use (See [here](https://guide.ncloud-docs.com/docs/clovastudio-explorer03#%ED%85%8C%EC%8A%A4%ED%8A%B8%EC%95%B1%EC%83%9D%EC%84%B1).)
4. Issue a Test or Service API key (See [here](https://guide.ncloud-docs.com/docs/clovastudio-explorer-testapp).)

### Credentials

Set the `CLOVASTUDIO_API_KEY` environment variable with your API key.
"""
logger.info("# ClovaXEmbeddings")

# import getpass

if not os.getenv("CLOVASTUDIO_API_KEY"):
#     os.environ["CLOVASTUDIO_API_KEY"] = getpass.getpass("Enter CLOVA Studio API Key: ")

"""
### Installation

ClovaXEmbeddings integration lives in the `langchain_naver` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-naver

"""
## Instantiation

Now we can instantiate our embeddings object and embed query or document:

- There are several embedding models available in CLOVA Studio. Please refer [here](https://guide.ncloud-docs.com/docs/en/clovastudio-explorer03#임베딩API) for further details.
- Note that you might need to normalize the embeddings depending on your specific use case.
"""
logger.info("## Instantiation")


embeddings = ClovaXEmbeddings(
    model="clir-emb-dolphin"  # set with the model name of corresponding test/service app. Default is `clir-emb-dolphin`
)

"""
## Indexing and Retrieval

Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/rag).

Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.
"""
logger.info("## Indexing and Retrieval")


text = "CLOVA Studio is an AI development tool that allows you to customize your own HyperCLOVA X models."

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

retrieved_documents = retriever.invoke("What is CLOVA Studio?")

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

text2 = "LangChain is a framework for building context-aware reasoning applications"
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    logger.debug(str(vector)[:100])  # Show the first 100 characters of the vector

"""
## API Reference

For detailed documentation on `ClovaXEmbeddings` features and configuration options, please refer to the [API reference](https://guide.ncloud-docs.com/docs/clovastudio-dev-langchain#%EC%9E%84%EB%B2%A0%EB%94%A9%EB%8F%84%EA%B5%AC%EC%9D%B4%EC%9A%A9).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)