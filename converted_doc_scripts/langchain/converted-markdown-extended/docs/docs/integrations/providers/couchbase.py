from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from datetime import timedelta
from jet.adapters.langchain.chat_ollama.Embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders.couchbase import CouchbaseLoader
from langchain_core.globals import set_llm_cache
from langchain_couchbase import CouchbaseSearchVectorStore
from langchain_couchbase.cache import CouchbaseCache
from langchain_couchbase.cache import CouchbaseSemanticCache
from langchain_couchbase.chat_message_histories import CouchbaseChatMessageHistory
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

>[Couchbase](http://couchbase.com/) is an award-winning distributed NoSQL cloud database
> that delivers unmatched versatility, performance, scalability, and financial value
> for all of your cloud, mobile, AI, and edge computing applications.

## Installation and Setup

We have to install the `langchain-couchbase` package.
"""
logger.info("# Couchbase")

pip install langchain-couchbase

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/couchbase).
"""
logger.info("## Vector Store")


# import getpass

# COUCHBASE_CONNECTION_STRING = getpass.getpass(
    "Enter the connection string for the Couchbase cluster: "
)
# DB_USERNAME = getpass.getpass("Enter the username for the Couchbase cluster: ")
# DB_PASSWORD = getpass.getpass("Enter the password for the Couchbase cluster: ")



auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
options = ClusterOptions(auth)
cluster = Cluster(COUCHBASE_CONNECTION_STRING, options)

cluster.wait_until_ready(timedelta(seconds=5))

vector_store = CouchbaseSearchVectorStore(
    cluster=cluster,
    bucket_name=BUCKET_NAME,
    scope_name=SCOPE_NAME,
    collection_name=COLLECTION_NAME,
    embedding=my_embeddings,
    index_name=SEARCH_INDEX_NAME,
)

texts = ["Couchbase is a NoSQL database", "LangChain is a framework for LLM applications"]
vectorstore.add_texts(texts)

query = "What is Couchbase?"
docs = vectorstore.similarity_search(query)

"""
API Reference: [CouchbaseSearchVectorStore](https://couchbase-ecosystem.github.io/langchain-couchbase/langchain_couchbase.html#module-langchain_couchbase.vectorstores.search_vector_store)

## Document loader

See a [usage example](/docs/integrations/document_loaders/couchbase).
"""
logger.info("## Document loader")


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

loader = CouchbaseLoader(
    connection_string,
    db_username,
    db_password,
    query,
)

docs = loader.load()

"""
## LLM Caches

### CouchbaseCache
Use Couchbase as a cache for prompts and responses.

See a [usage example](/docs/integrations/llm_caching/#couchbase-caches).

To import this cache:
"""
logger.info("## LLM Caches")


"""
To use this cache with your LLMs:
"""
logger.info("To use this cache with your LLMs:")


cluster = couchbase_cluster_connection_object

set_llm_cache(
    CouchbaseCache(
        cluster=cluster,
        bucket_name=BUCKET_NAME,
        scope_name=SCOPE_NAME,
        collection_name=COLLECTION_NAME,
    )
)

"""
API Reference: [CouchbaseCache](https://couchbase-ecosystem.github.io/langchain-couchbase/langchain_couchbase.html#langchain_couchbase.cache.CouchbaseCache)

### CouchbaseSemanticCache
Semantic caching allows users to retrieve cached prompts based on the semantic similarity between the user input and previously cached inputs. Under the hood it uses Couchbase as both a cache and a vectorstore.
The CouchbaseSemanticCache needs a Search Index defined to work. Please look at the [usage example](/docs/integrations/vectorstores/couchbase) on how to set up the index.

See a [usage example](/docs/integrations/llm_caching/#couchbase-caches).

To import this cache:
"""
logger.info("### CouchbaseSemanticCache")


"""
To use this cache with your LLMs:
"""
logger.info("To use this cache with your LLMs:")



embeddings = OllamaEmbeddings(model="mxbai-embed-large")
cluster = couchbase_cluster_connection_object

set_llm_cache(
    CouchbaseSemanticCache(
        cluster=cluster,
        embedding = embeddings,
        bucket_name=BUCKET_NAME,
        scope_name=SCOPE_NAME,
        collection_name=COLLECTION_NAME,
        index_name=INDEX_NAME,
    )
)

"""
API Reference: [CouchbaseSemanticCache](https://couchbase-ecosystem.github.io/langchain-couchbase/langchain_couchbase.html#langchain_couchbase.cache.CouchbaseSemanticCache)

## Chat Message History
Use Couchbase as the storage for your chat messages.

See a [usage example](/docs/integrations/memory/couchbase_chat_message_history).

To use the chat message history in your applications:
"""
logger.info("## Chat Message History")


message_history = CouchbaseChatMessageHistory(
    cluster=cluster,
    bucket_name=BUCKET_NAME,
    scope_name=SCOPE_NAME,
    collection_name=COLLECTION_NAME,
    session_id="test-session",
)

message_history.add_user_message("hi!")

"""
API Reference: [CouchbaseChatMessageHistory](https://couchbase-ecosystem.github.io/langchain-couchbase/langchain_couchbase.html#module-langchain_couchbase.chat_message_histories)
"""
logger.info("API Reference: [CouchbaseChatMessageHistory](https://couchbase-ecosystem.github.io/langchain-couchbase/langchain_couchbase.html#module-langchain_couchbase.chat_message_histories)")

logger.info("\n\n[DONE]", bright=True)