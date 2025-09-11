from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Lantern
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Tuple
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
# Lantern

>[Lantern](https://github.com/lanterndata/lantern) is an open-source vector similarity search for `Postgres`

It supports:
- Exact and approximate nearest neighbor search
- L2 squared distance, hamming distance, and cosine distance

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use the Postgres vector database (`Lantern`).

See the [installation instruction](https://github.com/lanterndata/lantern#-quick-install).

We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.

# Pip install necessary package
!pip install ollama
!pip install psycopg2-binary
!pip install tiktoken
"""
logger.info("# Lantern")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")



load_dotenv()


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# CONNECTION_STRING = getpass.getpass("DB Connection String:")

"""
## Similarity Search with Cosine Distance (Default)
"""
logger.info("## Similarity Search with Cosine Distance (Default)")

COLLECTION_NAME = "state_of_the_union_test"

db = Lantern.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True,
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

"""
## Maximal Marginal Relevance Search (MMR)
Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.
"""
logger.info("## Maximal Marginal Relevance Search (MMR)")

docs_with_score = db.max_marginal_relevance_search_with_score(query)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

"""
## Working with vectorstore

Above, we created a vectorstore from scratch. However, often times we want to work with an existing vectorstore.
In order to do that, we can initialize it directly.
"""
logger.info("## Working with vectorstore")

store = Lantern(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
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
This will delete the collection before re-populating it
"""
logger.info("### Overriding a vectorstore")

db = Lantern.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
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