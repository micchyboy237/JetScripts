from jet.logger import logger
from langchain_community.document_loaders.couchbase import CouchbaseLoader
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
# Couchbase

>[Couchbase](http://couchbase.com/) is an award-winning distributed NoSQL cloud database that delivers unmatched versatility, performance, scalability, and financial value for all of your cloud, mobile, AI, and edge computing applications.

## Installation
"""
logger.info("# Couchbase")

# %pip install --upgrade --quiet  couchbase

"""
## Querying for Documents from Couchbase
For more details on connecting to a Couchbase cluster, please check the [Python SDK documentation](https://docs.couchbase.com/python-sdk/current/howtos/managing-connections.html#connection-strings).

For help with querying for documents using SQL++ (SQL for JSON), please check the [documentation](https://docs.couchbase.com/server/current/n1ql/n1ql-language-reference/index.html).
"""
logger.info("## Querying for Documents from Couchbase")


connection_string = "couchbase://localhost"  # valid Couchbase connection string
db_username = (
    "Administrator"  # valid database user with read access to the bucket being queried
)
db_password = "Password"  # password for the database user

query = """
    SELECT h.* FROM `travel-sample`.inventory.hotel h
        WHERE h.country = 'United States'
        LIMIT 1
        """

"""
## Create the Loader
"""
logger.info("## Create the Loader")

loader = CouchbaseLoader(
    connection_string,
    db_username,
    db_password,
    query,
)

"""
You can fetch the documents by calling the `load` method of the loader. It will return a list with all the documents. If you want to avoid this blocking call, you can call `lazy_load` method that returns an Iterator.
"""
logger.info("You can fetch the documents by calling the `load` method of the loader. It will return a list with all the documents. If you want to avoid this blocking call, you can call `lazy_load` method that returns an Iterator.")

docs = loader.load()
logger.debug(docs)

docs_iterator = loader.lazy_load()
for doc in docs_iterator:
    logger.debug(doc)
    break

"""
## Specifying Fields with Content and Metadata
The fields that are part of the Document content can be specified using the `page_content_fields` parameter.
The metadata fields for the Document can be specified using the `metadata_fields` parameter.
"""
logger.info("## Specifying Fields with Content and Metadata")

loader_with_selected_fields = CouchbaseLoader(
    connection_string,
    db_username,
    db_password,
    query,
    page_content_fields=[
        "address",
        "name",
        "city",
        "phone",
        "country",
        "geo",
        "description",
        "reviews",
    ],
    metadata_fields=["id"],
)
docs_with_selected_fields = loader_with_selected_fields.load()
logger.debug(docs_with_selected_fields)

logger.info("\n\n[DONE]", bright=True)