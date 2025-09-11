from jet.adapters.langchain.chat_ollama import AzureOllamaEmbeddings
from jet.logger import logger
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
sidebar_label: AzureOllama
---

# AzureOllamaEmbeddings

This will help you get started with AzureOllama embedding models using LangChain. For detailed documentation on `AzureOllamaEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/ollama/embeddings/jet.adapters.langchain.chat_ollama.embeddings.azure.AzureOllamaEmbeddings.html).

## Overview
### Integration details


<ItemTable category="text_embedding" item="AzureOllama" />

## Setup

To access AzureOllama embedding models you'll need to create an Azure account, get an API key, and install the `langchain-ollama` integration package.

### Credentials

You’ll need to have an Azure Ollama instance deployed. You can deploy a version on Azure Portal following this [guide](https://learn.microsoft.com/en-us/azure/ai-services/ollama/how-to/create-resource?pivots=web-portal).

Once you have your instance running, make sure you have the name of your instance and key. You can find the key in the Azure Portal, under the “Keys and Endpoint” section of your instance.

```bash
AZURE_OPENAI_ENDPOINT=<YOUR API ENDPOINT>
# AZURE_OPENAI_API_KEY=<YOUR_KEY>
AZURE_OPENAI_API_VERSION="2024-02-01"
```
"""
logger.info("# AzureOllamaEmbeddings")

# import getpass

# if not os.getenv("AZURE_OPENAI_API_KEY"):
#     os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOllama API key: "
    )

"""
T
o
 
e
n
a
b
l
e
 
a
u
t
o
m
a
t
e
d
 
t
r
a
c
i
n
g
 
o
f
 
y
o
u
r
 
m
o
d
e
l
 
c
a
l
l
s
,
 
s
e
t
 
y
o
u
r
 
[
L
a
n
g
S
m
i
t
h
]
(
h
t
t
p
s
:
/
/
d
o
c
s
.
s
m
i
t
h
.
l
a
n
g
c
h
a
i
n
.
c
o
m
/
)
 
A
P
I
 
k
e
y
:
"""
logger.info("T")



"""
### Installation

The LangChain AzureOllama integration lives in the `langchain-ollama` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-ollama

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


embeddings = AzureOllamaEmbeddings(
    model="text-embedding-3-large",
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

For detailed documentation on `AzureOllamaEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/ollama/embeddings/jet.adapters.langchain.chat_ollama.embeddings.azure.AzureOllamaEmbeddings.html).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)