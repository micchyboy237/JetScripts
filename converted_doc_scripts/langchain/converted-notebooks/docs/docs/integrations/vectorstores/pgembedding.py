from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import PGEmbedding
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
# Postgres Embedding

> [Postgres Embedding](https://github.com/neondatabase/pg_embedding) is an open-source vector similarity search for `Postgres` that uses  `Hierarchical Navigable Small Worlds (HNSW)` for approximate nearest neighbor search.

>It supports:
>- exact and approximate nearest neighbor search using HNSW
>- L2 distance

This notebook shows how to use the Postgres vector database (`PGEmbedding`).

> The PGEmbedding integration creates the pg_embedding extension for you, but you run the following Postgres query to add it:
```sql
CREATE EXTENSION embedding;
```
"""
logger.info("# Postgres Embedding")

# %pip install --upgrade --quiet  langchain-ollama langchain-community
# %pip install --upgrade --quiet  psycopg2-binary
# %pip install --upgrade --quiet  tiktoken

"""
Add the Ollama API Key to the environment variables to use `OllamaEmbeddings`.
"""
logger.info("Add the Ollama API Key to the environment variables to use `OllamaEmbeddings`.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")



if "DATABASE_URL" not in os.environ:
#     os.environ["DATABASE_URL"] = getpass.getpass("Database Url:")

loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
connection_string = os.environ.get("DATABASE_URL")
collection_name = "state_of_the_union"

db = PGEmbedding.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=collection_name,
    connection_string=connection_string,
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score: List[Tuple[Document, float]] = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

"""
## Working with vectorstore in Postgres

### Uploading a vectorstore in PG
"""
logger.info("## Working with vectorstore in Postgres")

db = PGEmbedding.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=collection_name,
    connection_string=connection_string,
    pre_delete_collection=False,
)

"""
### Create HNSW Index
By default, the extension performs a sequential scan search, with 100% recall. You might consider creating an HNSW index for approximate nearest neighbor (ANN) search to speed up `similarity_search_with_score` execution time. To create the HNSW index on your vector column, use a `create_hnsw_index` function:
"""
logger.info("### Create HNSW Index")

PGEmbedding.create_hnsw_index(
    max_elements=10000, dims=1536, m=8, ef_construction=16, ef_search=16
)

"""
The function above is equivalent to running the below SQL query:
```sql
CREATE INDEX ON vectors USING hnsw(vec) WITH (maxelements=10000, dims=1536, m=3, efconstruction=16, efsearch=16);
```
The HNSW index options used in the statement above include:

- maxelements: Defines the maximum number of elements indexed. This is a required parameter. The example shown above has a value of 3. A real-world example would have a much large value, such as 1000000. An "element" refers to a data point (a vector) in the dataset, which is represented as a node in the HNSW graph. Typically, you would set this option to a value able to accommodate the number of rows in your in your dataset.
- dims: Defines the number of dimensions in your vector data. This is a required parameter. A small value is used in the example above. If you are storing data generated using Ollama's text-embedding-ada-002 model, which supports 1536 dimensions, you would define a value of 1536, for example.
- m: Defines the maximum number of bi-directional links (also referred to as "edges") created for each node during graph construction.
The following additional index options are supported:

- efConstruction: Defines the number of nearest neighbors considered during index construction. The default value is 32.
- efsearch: Defines the number of nearest neighbors considered during index search. The default value is 32.
For information about how you can configure these options to influence the HNSW algorithm, refer to [Tuning the HNSW algorithm](https://neon.tech/docs/extensions/pg_embedding#tuning-the-hnsw-algorithm).

### Retrieving a vectorstore in PG
"""
logger.info("### Retrieving a vectorstore in PG")

store = PGEmbedding(
    connection_string=connection_string,
    embedding_function=embeddings,
    collection_name=collection_name,
)

retriever = store.as_retriever()

retriever

db1 = PGEmbedding.from_existing_index(
    embedding=embeddings,
    collection_name=collection_name,
    pre_delete_collection=False,
    connection_string=connection_string,
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score: List[Tuple[Document, float]] = db1.similarity_search_with_score(query)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

logger.info("\n\n[DONE]", bright=True)