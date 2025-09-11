from dotenv import load_dotenv
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import (
    Kinetica,
    KineticaSettings,
)
from langchain_core.documents import Document
from uuid import uuid4
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
sidebar_label: Kinetica
---

# Kinetica Vectorstore API

>[Kinetica](https://www.kinetica.com/) is a database with integrated support for vector similarity search

It supports:
- exact and approximate nearest neighbor search
- L2 distance, inner product, and cosine distance

This notebook shows how to use the Kinetica vector store (`Kinetica`).

This needs an instance of Kinetica which can easily be setup using the instructions given here - [installation instruction](https://www.kinetica.com/developer-edition/).
"""
logger.info("# Kinetica Vectorstore API")

# %pip install --upgrade --quiet  langchain-ollama langchain-community
# %pip install gpudb>=7.2.2.0
# %pip install --upgrade --quiet  tiktoken

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info(
    "We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


load_dotenv()


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

HOST = os.getenv("KINETICA_HOST", "http://127.0.0.1:9191")
USERNAME = os.getenv("KINETICA_USERNAME", "")
PASSWORD = os.getenv("KINETICA_PASSWORD", "")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def create_config() -> KineticaSettings:
    return KineticaSettings(host=HOST, username=USERNAME, password=PASSWORD)


document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

"""
## Similarity Search with Euclidean Distance (Default)
"""
logger.info("## Similarity Search with Euclidean Distance (Default)")

COLLECTION_NAME = "langchain_example"
connection = create_config()

db = Kinetica(
    connection,
    embeddings,
    collection_name=COLLECTION_NAME,
)

db.add_documents(documents=documents, ids=uuids)


logger.debug()
logger.debug("Similarity Search")
results = db.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

logger.debug()
logger.debug("Similarity search with score")
results = db.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    logger.debug(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

"""
## Working with vectorstore

Above, we created a vectorstore from scratch. However, often times we want to work with an existing vectorstore.
In order to do that, we can initialize it directly.
"""
logger.info("## Working with vectorstore")

store = Kinetica(
    collection_name=COLLECTION_NAME,
    config=connection,
    embedding_function=embeddings,
)

"""
### Add documents
We can add documents to the existing vectorstore.
"""
logger.info("### Add documents")

store.add_documents([Document(page_content="foo")])

docs_with_score = db.similarity_search_with_score("foo")

docs_with_score[0]

docs_with_score[1]

"""
### Overriding a vectorstore

If you have an existing collection, you override it by doing `from_documents` and setting `pre_delete_collection` = True
"""
logger.info("### Overriding a vectorstore")

db = Kinetica.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    config=connection,
    pre_delete_collection=True,
)

docs_with_score = db.similarity_search_with_score("foo")

docs_with_score[0]

"""
### Using a VectorStore as a Retriever
"""
logger.info("### Using a VectorStore as a Retriever")

retriever = store.as_retriever()

logger.debug(retriever)

logger.info("\n\n[DONE]", bright=True)
