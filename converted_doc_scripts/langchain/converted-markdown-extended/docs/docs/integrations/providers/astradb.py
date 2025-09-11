from astrapy.info import (
CollectionLexicalOptions,
CollectionRerankOptions,
RerankServiceOptions,
VectorServiceOptions,
)
from astrapy.info import VectorServiceOptions
from jet.logger import logger
from langchain.globals import set_llm_cache
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_astradb import AstraDBByteStore
from langchain_astradb import AstraDBCache
from langchain_astradb import AstraDBChatMessageHistory
from langchain_astradb import AstraDBLoader
from langchain_astradb import AstraDBSemanticCache
from langchain_astradb import AstraDBStore
from langchain_astradb import AstraDBVectorStore
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
# Astra DB

> [DataStax Astra DB](https://docs.datastax.com/en/astra-db-serverless/index.html) is a serverless AI-ready database built on `Apache Cassandra®` and made conveniently available through an easy-to-use JSON API.

See a [tutorial provided by DataStax](https://docs.datastax.com/en/astra/astra-db-vector/tutorials/chatbot.html).

## Installation and Setup

Install the following Python package:
"""
logger.info("# Astra DB")

pip install "langchain-astradb>=0.6,<0.7"

"""
Create a database (if needed) and get the [connection secrets](https://docs.datastax.com/en/astra-db-serverless/get-started/quickstart.html#create-a-database-and-store-your-credentials).
Set the following variables:
"""
logger.info("Create a database (if needed) and get the [connection secrets](https://docs.datastax.com/en/astra-db-serverless/get-started/quickstart.html#create-a-database-and-store-your-credentials).")

ASTRA_DB_API_ENDPOINT="API_ENDPOINT"
ASTRA_DB_APPLICATION_TOKEN="TOKEN"

"""
## Vector Store

A few typical initialization patterns are shown here:
"""
logger.info("## Vector Store")


vector_store = AstraDBVectorStore(
    embedding=my_embedding,
    collection_name="my_store",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)



vector_store_vectorize = AstraDBVectorStore(
    collection_name="my_vectorize_store",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    collection_vector_service_options=VectorServiceOptions(
        provider="nvidia",
        model_name="NV-Embed-QA",
    ),
)



vector_store_hybrid = AstraDBVectorStore(
    collection_name="my_hybrid_store",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    collection_vector_service_options=VectorServiceOptions(
        provider="nvidia",
        model_name="NV-Embed-QA",
    ),
    collection_lexical=CollectionLexicalOptions(analyzer="standard"),
    collection_rerank=CollectionRerankOptions(
        service=RerankServiceOptions(
            provider="nvidia",
            model_name="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        ),
    ),
)

"""
Notable features of class `AstraDBVectorStore`:

- native async API;
- metadata filtering in search;
- MMR (maximum marginal relevance) search;
- server-side embedding computation (["vectorize"](https://docs.datastax.com/en/astra-db-serverless/databases/embedding-generation.html) in Astra DB parlance);
- auto-detect its settings from an existing, pre-populated Astra DB collection;
- [hybrid search](https://docs.datastax.com/en/astra-db-serverless/databases/hybrid-search.html#the-hybrid-search-process) (vector + BM25 and then a rerank step);
- support for non-Astra Data API (e.g. self-hosted [HCD](https://docs.datastax.com/en/hyper-converged-database/1.1/get-started/get-started-hcd.html) deployments);

Learn more in the [example notebook](/docs/integrations/vectorstores/astradb).

See the [example provided by DataStax](https://docs.datastax.com/en/astra/astra-db-vector/integrations/langchain.html).

## Chat message history
"""
logger.info("## Chat message history")


message_history = AstraDBChatMessageHistory(
    session_id="test-session",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

"""
See the [usage example](/docs/integrations/memory/astradb_chat_message_history#example).

## LLM Cache
"""
logger.info("## LLM Cache")


set_llm_cache(AstraDBCache(
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
))

"""
Learn more in the [example notebook](/docs/integrations/llm_caching#astra-db-caches) (scroll to the Astra DB section).


## Semantic LLM Cache
"""
logger.info("## Semantic LLM Cache")


set_llm_cache(AstraDBSemanticCache(
    embedding=my_embedding,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
))

"""
Learn more in the [example notebook](/docs/integrations/llm_caching#astra-db-caches) (scroll to the appropriate section).

## Document loader
"""
logger.info("## Document loader")


loader = AstraDBLoader(
    collection_name="my_collection",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

"""
Learn more in the [example notebook](/docs/integrations/document_loaders/astradb).

## Self-querying retriever
"""
logger.info("## Self-querying retriever")


vector_store = AstraDBVectorStore(
    embedding=my_embedding,
    collection_name="my_store",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

retriever = SelfQueryRetriever.from_llm(
    my_llm,
    vector_store,
    document_content_description,
    metadata_field_info
)

"""
Learn more in the [example notebook](/docs/integrations/retrievers/self_query/astradb).

## Store
"""
logger.info("## Store")


store = AstraDBStore(
    collection_name="my_kv_store",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

"""
See the API Reference for the [AstraDBStore](https://python.langchain.com/api_reference/astradb/storage/langchain_astradb.storage.AstraDBStore.html).

## Byte Store
"""
logger.info("## Byte Store")


store = AstraDBByteStore(
    collection_name="my_kv_store",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

"""
See the API reference for the [AstraDBByteStore](https://python.langchain.com/api_reference/astradb/storage/langchain_astradb.storage.AstraDBByteStore.html).
"""
logger.info("See the API reference for the [AstraDBByteStore](https://python.langchain.com/api_reference/astradb/storage/langchain_astradb.storage.AstraDBByteStore.html).")

logger.info("\n\n[DONE]", bright=True)