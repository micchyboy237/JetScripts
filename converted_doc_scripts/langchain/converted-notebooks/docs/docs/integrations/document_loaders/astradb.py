from jet.logger import logger
from langchain_astradb import AstraDBLoader
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
# AstraDB

> [DataStax Astra DB](https://docs.datastax.com/en/astra-db-serverless/index.html) is a serverless 
> AI-ready database built on `Apache Cassandra®` and made conveniently available 
> through an easy-to-use JSON API.

## Overview

The Astra DB Document Loader returns a list of Langchain `Document` objects read from an Astra DB collection.

The loader takes the following parameters:

* `api_endpoint`: Astra DB API endpoint. Looks like `https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com`
* `token`: Astra DB token. Looks like `AstraCS:aBcD0123...`
* `collection_name` : AstraDB collection name
* `namespace`: (Optional) AstraDB namespace (called _keyspace_ in Astra DB)
* `filter_criteria`: (Optional) Filter used in the find query
* `projection`: (Optional) Projection used in the find query
* `limit`: (Optional) Maximum number of documents to retrieve
* `extraction_function`: (Optional) A function to convert the AstraDB document to the LangChain `page_content` string. Defaults to `json.dumps`

The loader sets the following metadata for the documents it reads:

```python
metadata={
    "namespace": "...", 
    "api_endpoint": "...", 
    "collection": "..."
}
```

## Setup
"""
logger.info("# AstraDB")

# !pip install "langchain-astradb>=0.6,<0.7"

"""
## Load documents with the Document Loader
"""
logger.info("## Load documents with the Document Loader")


"""
[**API Reference:** `AstraDBLoader`](https://python.langchain.com/api_reference/astradb/document_loaders/langchain_astradb.document_loaders.AstraDBLoader.html#langchain_astradb.document_loaders.AstraDBLoader)
"""

# from getpass import getpass

ASTRA_DB_API_ENDPOINT = input("ASTRA_DB_API_ENDPOINT = ")
# ASTRA_DB_APPLICATION_TOKEN = getpass("ASTRA_DB_APPLICATION_TOKEN = ")

loader = AstraDBLoader(
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    collection_name="movie_reviews",
    projection={"title": 1, "reviewtext": 1},
    limit=10,
)

docs = loader.load()

docs[0]

logger.info("\n\n[DONE]", bright=True)