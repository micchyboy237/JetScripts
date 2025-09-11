from jet.logger import logger
from langchain_community.document_loaders.surrealdb import SurrealDBLoader
import json
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
# SurrealDB

>[SurrealDB](https://surrealdb.com/) is an end-to-end cloud-native database designed for modern applications, including web, mobile, serverless, Jamstack, backend, and traditional applications. With SurrealDB, you can simplify your database and API infrastructure, reduce development time, and build secure, performant apps quickly and cost-effectively.
>
>**Key features of SurrealDB include:**
>
>* **Reduces development time:** SurrealDB simplifies your database and API stack by removing the need for most server-side components, allowing you to build secure, performant apps faster and cheaper.
>* **Real-time collaborative API backend service:** SurrealDB functions as both a database and an API backend service, enabling real-time collaboration.
>* **Support for multiple querying languages:** SurrealDB supports SQL querying from client devices, GraphQL, ACID transactions, WebSocket connections, structured and unstructured data, graph querying, full-text indexing, and geospatial querying.
>* **Granular access control:** SurrealDB provides row-level permissions-based access control, giving you the ability to manage data access with precision.
>
>View the [features](https://surrealdb.com/features), the latest [releases](https://surrealdb.com/releases), and [documentation](https://surrealdb.com/docs).

This notebook shows how to use functionality related to the `SurrealDBLoader`.

## Overview

The SurrealDB Document Loader returns a list of Langchain Documents from a SurrealDB database.

The Document Loader takes the following optional parameters:

* `dburl`: connection string to the websocket endpoint. default: `ws://localhost:8000/rpc`
* `ns`: name of the namespace. default: `langchain`
* `db`: name of the database. default: `database`
* `table`: name of the table. default: `documents`
* `db_user`: SurrealDB credentials if needed: db username.
* `db_pass`: SurrealDB credentails if needed: db password.
* `filter_criteria`: dictionary to construct the `WHERE` clause for filtering results from table.

The output `Document` takes the following shape:
```
Document(
    page_content=<json encoded string containing the result document>,
    metadata={
        'id': <document id>,
        'ns': <namespace name>,
        'db': <database_name>,
        'table': <table name>,
        ... <additional fields from metadata property of the document>
    }
)
```

## Setup

Uncomment the below cells to install surrealdb and langchain.
"""
logger.info("# SurrealDB")



# import nest_asyncio

# nest_asyncio.apply()



loader = SurrealDBLoader(
    dburl="ws://localhost:8000/rpc",
    ns="langchain",
    db="database",
    table="documents",
    db_user="root",
    db_pass="root",
    filter_criteria={},
)
docs = loader.load()
len(docs)

doc = docs[-1]
doc.metadata

len(doc.page_content)

page_content = json.loads(doc.page_content)

page_content["text"]

logger.info("\n\n[DONE]", bright=True)