from jet.transformers.formatters import format_json
from google.colab import auth
from jet.logger import logger
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_bigtable import (
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from typing import List, Optional, Any, Union
import json
import os
import shutil

async def main():
    BigtableByteStore,
    BigtableEngine,
    init_key_value_store_table,
    )
    
    
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
    sidebar_label: Google Bigtable
    ---
    
    # BigtableByteStore
    
    This guide covers how to use Google Cloud Bigtable as a key-value store.
    
    [Bigtable](https://cloud.google.com/bigtable) is a key-value and wide-column store, ideal for fast access to structured, semi-structured, or unstructured data. 
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-bigtable-python/blob/main/docs/key_value_store.ipynb)
    
    ## Overview
    
    The `BigtableByteStore` uses Google Cloud Bigtable as a backend for a key-value store. It supports synchronous and asynchronous operations for setting, getting, and deleting key-value pairs.
    
    ### Integration details
    | Class | Package | Local | JS support | Package downloads | Package latest |
    | :--- | :--- | :---: | :---: | :---: | :---: |
    | [BigtableByteStore](https://github.com/googleapis/langchain-google-bigtable-python/blob/main/src/langchain_google_bigtable/key_value_store.py) | [langchain-google-bigtable](https://pypi.org/project/langchain-google-bigtable/) | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-google-bigtable?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-bigtable) |
    
    ## Setup
    
    ### Prerequisites
    
    To get started, you will need a Google Cloud project with an active Bigtable instance and table. 
    * [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
    * [Enable the Bigtable API](https://console.cloud.google.com/flows/enableapi?apiid=bigtable.googleapis.com)
    * [Create a Bigtable instance and table](https://cloud.google.com/bigtable/docs/creating-instance)
    
    ### Installation
    
    The integration is in the `langchain-google-bigtable` package. The command below also installs `langchain-google-vertexai` for the embedding cache example.
    """
    logger.info("# BigtableByteStore")
    
    # %pip install -qU langchain-google-bigtable langchain-google-vertexai
    
    """
    ### ☁ Set Your Google Cloud Project
    Set your Google Cloud project to use its resources within this notebook.
    
    If you don't know your project ID, you can run `gcloud config list` or see the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113).
    """
    logger.info("### ☁ Set Your Google Cloud Project")
    
    PROJECT_ID = "your-gcp-project-id"  # @param {type:"string"}
    INSTANCE_ID = "your-instance-id"  # @param {type:"string"}
    TABLE_ID = "your-table-id"  # @param {type:"string"}
    
    # !gcloud config set project {PROJECT_ID}
    
    """
    ### 🔐 Authentication
    Authenticate to Google Cloud to access your project resources.
    - For **Colab**, use the cell below.
    - For **Vertex AI Workbench**, see the [setup instructions](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).
    """
    logger.info("### 🔐 Authentication")
    
    
    auth.authenticate_user()
    
    """
    ## Instantiation
    
    To use `BigtableByteStore`, we first ensure a table exists and then initialize a `BigtableEngine` to manage connections.
    """
    logger.info("## Instantiation")
    
    
    init_key_value_store_table(
        project_id=PROJECT_ID,
        instance_id=INSTANCE_ID,
        table_id=TABLE_ID,
    )
    
    """
    ### BigtableEngine
    A `BigtableEngine` object handles the execution context for the store, especially for async operations. It's recommended to initialize a single engine and reuse it across multiple stores for better performance.
    """
    logger.info("### BigtableEngine")
    
    engine = await BigtableEngine.async_initialize(
            project_id=PROJECT_ID, instance_id=INSTANCE_ID
        )
    logger.success(format_json(engine))
    
    """
    ### BigtableByteStore
    
    This is the main class for interacting with the key-value store. It provides the methods for setting, getting, and deleting data.
    """
    logger.info("### BigtableByteStore")
    
    store = await BigtableByteStore.create(engine=engine, table_id=TABLE_ID)
    logger.success(format_json(store))
    
    """
    ## Usage
    
    The store supports both sync (`mset`, `mget`) and async (`amset`, `amget`) methods. This guide uses the async versions.
    
    ### Set
    Use `amset` to save key-value pairs to the store.
    """
    logger.info("## Usage")
    
    kv_pairs = [
        ("key1", b"value1"),
        ("key2", b"value2"),
        ("key3", b"value3"),
    ]
    
    await store.amset(kv_pairs)
    
    """
    ### Get
    Use `amget` to retrieve values. If a key is not found, `None` is returned for that key.
    """
    logger.info("### Get")
    
    retrieved_vals = await store.amget(["key1", "key2", "nonexistent_key"])
    logger.success(format_json(retrieved_vals))
    logger.debug(retrieved_vals)
    
    """
    ### Delete
    Use `amdelete` to remove keys from the store.
    """
    logger.info("### Delete")
    
    await store.amdelete(["key3"])
    
    await store.amget(["key1", "key3"])
    
    """
    ### Iterate over keys
    Use `ayield_keys` to iterate over all keys or keys with a specific prefix.
    """
    logger.info("### Iterate over keys")
    
    all_keys = [key async for key in store.ayield_keys()]
    logger.debug(f"All keys: {all_keys}")
    
    prefixed_keys = [key async for key in store.ayield_keys(prefix="key1")]
    logger.debug(f"Prefixed keys: {prefixed_keys}")
    
    """
    ## Advanced Usage: Embedding Caching
    
    A common use case for a key-value store is to cache expensive operations like computing text embeddings, which saves time and cost.
    """
    logger.info("## Advanced Usage: Embedding Caching")
    
    
    underlying_embeddings = VertexAIEmbeddings(
        project=PROJECT_ID, model_name="textembedding-gecko@003"
    )
    
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace="text-embeddings"
    )
    
    logger.debug("First call (computes and caches embedding):")
    # %time embedding_result_1 = await cached_embedder.aembed_query("Hello, world!")
    logger.success(format_json(# %time embedding_result_1))
    
    logger.debug("\nSecond call (retrieves from cache):")
    # %time embedding_result_2 = await cached_embedder.aembed_query("Hello, world!")
    logger.success(format_json(# %time embedding_result_2))
    
    """
    ### As a Simple Document Retriever
    
    This section shows how to create a simple retriever using the Bigtable store. It acts as a document persistence layer, fetching documents that match a query prefix.
    """
    logger.info("### As a Simple Document Retriever")
    
    
    
    class SimpleKVStoreRetriever(BaseRetriever):
        """A simple retriever that retrieves documents based on a prefix match in the key-value store."""
    
        store: BigtableByteStore
        documents: List[Union[Document, str]]
        k: int
    
        def set_up_store(self):
            kv_pairs_to_set = []
            for i, doc in enumerate(self.documents):
                if isinstance(doc, str):
                    doc = Document(page_content=doc)
                if not doc.id:
                    doc.id = str(i)
                value = (
                    "Page Content\n"
                    + doc.page_content
                    + "\nMetadata"
                    + json.dumps(doc.metadata)
                )
                kv_pairs_to_set.append((doc.id, value.encode("utf-8")))
            self.store.mset(kv_pairs_to_set)
    
        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            keys = [key async for key in self.store.ayield_keys(prefix=query)][: self.k]
            documents_retrieved = []
            async for document in await self.store.amget(keys):
                if document:
                    document_str = document.decode("utf-8")
                    page_content = document_str.split("Content\n")[1].split("\nMetadata")[0]
                    metadata = json.loads(document_str.split("\nMetadata")[1])
                    documents_retrieved.append(
                        Document(page_content=page_content, metadata=metadata)
                    )
            return documents_retrieved
    
        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> list[Document]:
            keys = [key for key in self.store.yield_keys(prefix=query)][: self.k]
            documents_retrieved = []
            for document in self.store.mget(keys):
                if document:
                    document_str = document.decode("utf-8")
                    page_content = document_str.split("Content\n")[1].split("\nMetadata")[0]
                    metadata = json.loads(document_str.split("\nMetadata")[1])
                    documents_retrieved.append(
                        Document(page_content=page_content, metadata=metadata)
                    )
            return documents_retrieved
    
    documents = [
        Document(
            page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
            metadata={"type": "fish", "trait": "low maintenance"},
            id="fish#Goldfish",
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"type": "cat", "trait": "independence"},
            id="mammals#Cats",
        ),
        Document(
            page_content="Rabbits are social animals that need plenty of space to hop around.",
            metadata={"type": "rabbit", "trait": "social"},
            id="mammals#Rabbits",
        ),
    ]
    
    retriever_store = BigtableByteStore.create_sync(
        engine=engine, instance_id=INSTANCE_ID, table_id=TABLE_ID
    )
    
    KVDocumentRetriever = SimpleKVStoreRetriever(
        store=retriever_store, documents=documents, k=2
    )
    
    KVDocumentRetriever.set_up_store()
    
    KVDocumentRetriever.invoke("fish")
    
    KVDocumentRetriever.invoke("mammals")
    
    """
    ## API reference
    
    For full details on the `BigtableByteStore` class, see the source code on [GitHub](https://github.com/googleapis/langchain-google-bigtable-python/blob/main/src/langchain_google_bigtable/key_value_store.py).
    """
    logger.info("## API reference")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())