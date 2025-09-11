from jet.logger import logger
import os
import shutil
import {CategoryTable, IndexTable} from '@theme/FeatureTables'


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
sidebar_position: 0
sidebar_class_name: hidden
---


# Retrievers

A [retriever](/docs/concepts/retrievers) is an interface that returns documents given an unstructured query.
It is more general than a vector store.
A retriever does not need to be able to store documents, only to return (or retrieve) them.
Retrievers can be created from vector stores, but are also broad enough to include [Wikipedia search](/docs/integrations/retrievers/wikipedia/) and [Amazon Kendra](/docs/integrations/retrievers/amazon_kendra_retriever/).

Retrievers accept a string query as input and return a list of [Documents](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) as output.

For specifics on how to use retrievers, see the [relevant how-to guides here](/docs/how_to/#retrievers).

Note that all [vector stores](/docs/concepts/vectorstores) can be [cast to retrievers](/docs/how_to/vectorstore_retriever/).
Refer to the vector store [integration docs](/docs/integrations/vectorstores/) for available vector stores.
This page lists custom retrievers, implemented via subclassing [BaseRetriever](/docs/how_to/custom_retriever/).

## Bring-your-own documents

The below retrievers allow you to index and search a custom corpus of documents.

<CategoryTable category="document_retrievers" />

## External index

The below retrievers will search over an external index (e.g., constructed from Internet data or similar).

<CategoryTable category="external_retrievers" />

## All retrievers

> **Note:** The descriptions in the table below are truncated for readability.

<IndexTable />
"""
logger.info("# Retrievers")

logger.info("\n\n[DONE]", bright=True)