from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.duckdb import DuckDBVectorStore
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
# DuckDB

>[DuckDB](https://duckdb.org/docs/api/python/overview) is a fast in-process analytical database. DuckDB is under an MIT license.

In this notebook we are going to show how to use DuckDB as a Vector store to be used in LlamaIndex.

Install DuckDB with:

```sh
pip install duckdb
```

Make sure to use the latest DuckDB version (>= 0.10.0).

You can run DuckDB in different modes depending on persistence:
- `in-memory` is the default mode, where the database is created in memory, you can force this to be use by setting `database_name = ":memory:"` when initializing the vector store.
- `persistence` is set by using a name for a database and setting a persistence directory `database_name = "my_vector_store.duckdb"` where the database is persisted in the default `persist_dir` or to the one you set it to.

With the vector store created, you can:
- `.add` 
- `.get` 
- `.update`
- `.upsert`
- `.delete`
- `.peek`
- `.query` to run a search.

## Basic example

In this basic example, we take the Paul Graham essay, split it into chunks, embed it using an open-source embedding model, load it into `DuckDBVectorStore`, and then query it.

For the embedding model we will use MLX.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# DuckDB")

# !pip install llama-index

"""
### Creating a DuckDB Index
"""
logger.info("### Creating a DuckDB Index")

# !pip install duckdb
# !pip install llama-index-vector-stores-duckdb




# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
Download and prepare the sample dataset
"""
logger.info("Download and prepare the sample dataset")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader(f"{GENERATED_DIR}/paul_graham/").load_data()

vector_store = DuckDBVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
## Persisting to disk example

Extending the previous example, if you want to save to disk, simply initialize the DuckDBVectorStore by specifying a database name and persist directory.
"""
logger.info("## Persisting to disk example")

documents = SimpleDirectoryReader(f"{GENERATED_DIR}/paul_graham/").load_data()

vector_store = DuckDBVectorStore("pg.duckdb", persist_dir="./persist/")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

vector_store = DuckDBVectorStore.from_local("./persist/pg.duckdb")
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
## Metadata filter example

It is possible to narrow down the search space by filter with metadata. Below is an example to show that in practice.
"""
logger.info("## Metadata filter example")


nodes = [
    TextNode(
        **{
            "text": "The Shawshank Redemption",
            "metadata": {
                "author": "Stephen King",
                "theme": "Friendship",
                "year": 1994,
                "ref_doc_id": "doc_1",
            },
        }
    ),
    TextNode(
        **{
            "text": "The Godfather",
            "metadata": {
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
                "year": 1972,
                "ref_doc_id": "doc_1",
            },
        }
    ),
    TextNode(
        **{
            "text": "Inception",
            "metadata": {
                "director": "Christopher Nolan",
                "theme": "Sci-fi",
                "year": 2010,
                "ref_doc_id": "doc_2",
            },
        }
    ),
]

vector_store = DuckDBVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
Define the metadata filters.
"""
logger.info("Define the metadata filters.")


filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

"""
Use the index as a retriever to use the metadatafilter option.
"""
logger.info("Use the index as a retriever to use the metadatafilter option.")

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

logger.info("\n\n[DONE]", bright=True)