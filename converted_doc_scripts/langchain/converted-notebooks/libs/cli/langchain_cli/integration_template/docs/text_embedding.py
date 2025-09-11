from __module_name__ import __ModuleName__Embeddings
from jet.logger import logger
from langchain_core.vectorstores import InMemoryVectorStore
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
sidebar_label: __ModuleName__
---

# __ModuleName__Embeddings

- [ ] TODO: Make sure API reference link is correct

This will help you get started with __ModuleName__ embedding models using LangChain. For detailed documentation on `__ModuleName__Embeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/v0.2/api_reference/__package_name_short__/embeddings/__module_name__.embeddings__ModuleName__Embeddings.html).

## Overview
### Integration details

| Provider | Package |
|:--------:|:-------:|
| [__ModuleName__](/docs/integrations/providers/__package_name_short__/) | [__package_name__](https://python.langchain.com/v0.2/api_reference/__module_name__/embeddings/__module_name__.embeddings__ModuleName__Embeddings.html) |

## Setup

- [ ] TODO: Update with relevant info.

To access __ModuleName__ embedding models you'll need to create a/an __ModuleName__ account, get an API key, and install the `__package_name__` integration package.

### Credentials

- TODO: Update with relevant info.

Head to (TODO: link) to sign up to __ModuleName__ and generate an API key. Once you've done this set the __MODULE_NAME___API_KEY environment variable:
"""
logger.info("# __ModuleName__Embeddings")

# import getpass

if not os.getenv("__MODULE_NAME___API_KEY"):
#     os.environ["__MODULE_NAME___API_KEY"] = getpass.getpass("Enter your __ModuleName__ API key: ")

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

The LangChain __ModuleName__ integration lives in the `__package_name__` package:
"""
logger.info("### Installation")

# %pip install -qU __package_name__

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:

- TODO: Update model instantiation with relevant params.
"""
logger.info("## Instantiation")


embeddings = __ModuleName__Embeddings(
    model="model-name",
)

"""
## Indexing and Retrieval

Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/).

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
logger.debug(str(single_vector)[:100]) # Show the first 100 characters of the vector

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
    logger.debug(str(vector)[:100]) # Show the first 100 characters of the vector

"""
## API Reference

For detailed documentation on `__ModuleName__Embeddings` features and configuration options, please refer to the [API reference](https://api.python.langchain.com/en/latest/embeddings/__module_name__.embeddings.__ModuleName__Embeddings.html).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)