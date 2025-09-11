from jet.logger import logger
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
# Key-value stores

## Overview

LangChain provides a key-value store interface for storing and retrieving data.

LangChain includes a [`BaseStore`](https://python.langchain.com/api_reference/core/stores/langchain_core.stores.BaseStore.html) interface,
which allows for storage of arbitrary data. However, LangChain components that require KV-storage accept a
more specific `BaseStore[str, bytes]` instance that stores binary data (referred to as a `ByteStore`), and internally take care of
encoding and decoding data for their specific needs.

This means that as a user, you only need to think about one type of store rather than different ones for different types of data.

## Usage

The key-value store interface in LangChain is used primarily for:

1. Caching [embeddings](/docs/concepts/embedding_models) via [CachedBackedEmbeddings](https://python.langchain.com/api_reference/langchain/embeddings/langchain.embeddings.cache.CacheBackedEmbeddings.html#langchain.embeddings.cache.CacheBackedEmbeddings) to avoid recomputing embeddings for repeated queries or when re-indexing content.

2. As a simple [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document) persistence layer in some retrievers.

Please see these how-to guides for more information:

* [How to cache embeddings guide](/docs/how_to/caching_embeddings/).
* [How to retriever using multiple vectors per document](/docs/how_to/custom_retriever/).

## Interface

All [`BaseStores`](https://python.langchain.com/api_reference/core/stores/langchain_core.stores.BaseStore.html) support the following interface. Note that the interface allows for modifying **multiple** key-value pairs at once:

- `mget(key: Sequence[str]) -> List[Optional[bytes]]`: get the contents of multiple keys, returning `None` if the key does not exist
- `mset(key_value_pairs: Sequence[Tuple[str, bytes]]) -> None`: set the contents of multiple keys
- `mdelete(key: Sequence[str]) -> None`: delete multiple keys
- `yield_keys(prefix: Optional[str] = None) -> Iterator[str]`: yield all keys in the store, optionally filtering by a prefix

## Integrations

Please reference the [stores integration page](/docs/integrations/stores/) for a list of available key-value store integrations.
"""
logger.info("# Key-value stores")

logger.info("\n\n[DONE]", bright=True)