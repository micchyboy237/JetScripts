from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_core.documents import Document
from langchain_mariadb import MariaDBStore
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
# MariaDB

LangChain's MariaDB integration (langchain-mariadb) provides vector capabilities for working with MariaDB version 11.7.1 and above, distributed under the MIT license. Users can use the provided implementations as-is or customize them for specific needs.
 Key features include:

 * Built-in vector similarity search
 * Support for cosine and euclidean distance metrics
 * Robust metadata filtering options
 * Performance optimization through connection pooling
 * Configurable table and column settings

## Setup

Launch a MariaDB Docker container with:
"""
logger.info("# MariaDB")

# !docker run --name mariadb-container -e MARIADB_ROOT_PASSWORD=langchain -e MARIADB_DATABASE=langchain -p 3306:3306 -d mariadb:11.7

"""
### Installing the Package

The package uses SQLAlchemy but works best with the MariaDB connector, which requires C/C++ components:
"""
logger.info("### Installing the Package")

# !sudo apt install libmariadb3 libmariadb-dev

# !sudo yum install MariaDB-shared MariaDB-devel

# !pip install -U mariadb

"""
Then install `langchain-mariadb` package
"""
logger.info("Then install `langchain-mariadb` package")

pip install -U langchain-mariadb

"""
VectorStore works along with an LLM model, here using `langchain-ollama` as example.
"""
logger.info("VectorStore works along with an LLM model, here using `langchain-ollama` as example.")

pip install langchain-ollama
# export OPENAI_API_KEY=...

"""
## Initialization
"""
logger.info("## Initialization")


url = f"mariadb+mariadbconnector://myuser:mypassword@localhost/langchain"

vectorstore = MariaDBStore(
    embeddings=OllamaEmbeddings(model="mxbai-embed-large"),
    embedding_length=1536,
    datasource=url,
    collection_name="my_docs",
)

"""
## Manage vector store

### Adding Data
You can add data as documents with metadata:
"""
logger.info("## Manage vector store")

docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
]
vectorstore.add_documents(docs)

"""
Or as plain text with optional metadata:
"""
logger.info("Or as plain text with optional metadata:")

texts = [
    "a sculpture exhibit is also at the museum",
    "a new coffee shop opened on Main Street",
]
metadatas = [
    {"id": 6, "location": "museum", "topic": "art"},
    {"id": 7, "location": "Main Street", "topic": "food"},
]

vectorstore.add_texts(texts=texts, metadatas=metadatas)

"""
## Query vector store
"""
logger.info("## Query vector store")

results = vectorstore.similarity_search("Hello", k=2)

results = vectorstore.similarity_search("Hello", filter={"category": "greeting"})

"""
### Filter Options

The system supports various filtering operations on metadata:

* Equality: $eq
* Inequality: $ne
* Comparisons: $lt, $lte, $gt, $gte
* List operations: $in, $nin
* Text matching: $like, $nlike
* Logical operations: $and, $or, $not

Example:
"""
logger.info("### Filter Options")

results = vectorstore.similarity_search(
    "kitty", k=10, filter={"id": {"$in": [1, 5, 2, 9]}}
)

results = vectorstore.similarity_search(
    "ducks",
    k=10,
    filter={"id": {"$in": [1, 5, 2, 9]}, "location": {"$in": ["pond", "market"]}},
)

"""
## Usage for retrieval-augmented generation

TODO: document example

## API reference

See the repo [here](https://github.com/mariadb-corporation/langchain-mariadb) for more detail.
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)