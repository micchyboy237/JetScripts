from __module_name__.vectorstores import __ModuleName__VectorStore
from jet.logger import logger
from langchain_core.documents import Document
import EmbeddingTabs from "@theme/EmbeddingTabs";
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

# __ModuleName__VectorStore

This notebook covers how to get started with the __ModuleName__ vector store.

## Setup

- TODO: Update with relevant info.
- TODO: Update minimum version to be correct.

To access __ModuleName__ vector stores you'll need to create a/an __ModuleName__ account, get an API key, and install the `__package_name__` integration package.

%pip install -qU "__package_name__>=MINIMUM_VERSION"

### Credentials

- TODO: Update with relevant info.

Head to (TODO: link) to sign up to __ModuleName__ and generate an API key. Once you've done this set the __MODULE_NAME___API_KEY environment variable:
"""
logger.info("# __ModuleName__VectorStore")

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
## Initialization

- TODO: Fill out with relevant init params


```{=mdx}

<EmbeddingTabs/>
```
"""
logger.info("## Initialization")


vector_store = __ModuleName__VectorStore(embeddings=embeddings)

"""
## Manage vector store

### Add items to vector store

- TODO: Edit and then run code cell to generate output
"""
logger.info("## Manage vector store")


document_1 = Document(
    page_content="foo",
    metadata={"source": "https://example.com"}
)

document_2 = Document(
    page_content="bar",
    metadata={"source": "https://example.com"}
)

document_3 = Document(
    page_content="baz",
    metadata={"source": "https://example.com"}
)

documents = [document_1, document_2, document_3]

vector_store.add_documents(documents=documents,ids=["1","2","3"])

"""
### Update items in vector store

- TODO: Edit and then run code cell to generate output
"""
logger.info("### Update items in vector store")

updated_document = Document(
    page_content="qux",
    metadata={"source": "https://another-example.com"}
)

vector_store.update_documents(document_id="1",document=updated_document)

"""
### Delete items from vector store

- TODO: Edit and then run code cell to generate output
"""
logger.info("### Delete items from vector store")

vector_store.delete(ids=["3"])

"""
## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

### Query directly

Performing a simple similarity search can be done as follows:

- TODO: Edit and then run code cell to generate output
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(query="thud",k=1,filter={"source":"https://another-example.com"})
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

"""
If you want to execute a similarity search and receive the corresponding scores you can run:

- TODO: Edit and then run code cell to generate output
"""
logger.info("If you want to execute a similarity search and receive the corresponding scores you can run:")

results = vector_store.similarity_search_with_score(query="thud",k=1,filter={"source":"https://example.com"})
for doc, score in results:
    logger.debug(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

"""
### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.

- TODO: Edit and then run code cell to generate output
"""
logger.info("### Query by turning into retriever")

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1}
)
retriever.invoke("thud")

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval/)

## TODO: Any functionality specific to this vector store

E.g. creating a persisten database to save to your disk, etc.

## API reference

For detailed documentation of all __ModuleName__VectorStore features and configurations head to the API reference: https://api.python.langchain.com/en/latest/vectorstores/__module_name__.vectorstores.__ModuleName__VectorStore.html
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)