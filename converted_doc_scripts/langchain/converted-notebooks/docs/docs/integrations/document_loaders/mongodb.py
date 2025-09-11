from jet.logger import logger
from langchain_community.document_loaders.mongodb import MongodbLoader
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
# MongoDB

[MongoDB](https://www.mongodb.com/) is a NoSQL , document-oriented database that supports JSON-like documents with a dynamic schema.

## Overview

The MongoDB Document Loader returns a list of Langchain Documents from a MongoDB database.

The Loader requires the following parameters:

*   MongoDB connection string
*   MongoDB database name
*   MongoDB collection name
*   (Optional) Content Filter dictionary
*   (Optional) List of field names to include in the output

The output takes the following format:

- pageContent= Mongo Document
- metadata=\{'database': '[database_name]', 'collection': '[collection_name]'\}

## Load the Document Loader
"""
logger.info("# MongoDB")

# import nest_asyncio

# nest_asyncio.apply()


loader = MongodbLoader(
    connection_string="mongodb://localhost:27017/",
    db_name="sample_restaurants",
    collection_name="restaurants",
    filter_criteria={"borough": "Bronx", "cuisine": "Bakery"},
    field_names=["name", "address"],
)

docs = loader.load()

len(docs)

docs[0]

logger.info("\n\n[DONE]", bright=True)