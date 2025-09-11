from azure.cosmos import CosmosClient, PartitionKey
from cassandra.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from datetime import timedelta
from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.adapters.langchain.chat_ollama import Ollama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.cache import SQLAlchemyCache
from langchain.chains.summarize import load_summarize_chain
from langchain.globals import set_llm_cache
from langchain_astradb import AstraDBCache
from langchain_astradb import AstraDBSemanticCache
from langchain_community.cache import AzureCosmosDBNoSqlSemanticCache
from langchain_community.cache import AzureCosmosDBSemanticCache
from langchain_community.cache import CassandraCache
from langchain_community.cache import CassandraSemanticCache
from langchain_community.cache import GPTCache
from langchain_community.cache import InMemoryCache
from langchain_community.cache import MemcachedCache
from langchain_community.cache import MomentoCache
from langchain_community.cache import OpenSearchSemanticCache
from langchain_community.cache import RedisCache
from langchain_community.cache import RedisSemanticCache
from langchain_community.cache import SQLAlchemyCache
from langchain_community.cache import SQLiteCache
from langchain_community.cache import UpstashRedisCache
from langchain_community.vectorstores.azure_cosmos_db import (
CosmosDBSimilarityType,
CosmosDBVectorSearchType,
)
from langchain_core.caches import RETURN_VAL_TYPE
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_couchbase.cache import CouchbaseCache
from langchain_couchbase.cache import CouchbaseSemanticCache
from langchain_elasticsearch import ElasticsearchCache
from langchain_elasticsearch import ElasticsearchEmbeddingsCache
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from langchain_mongodb.cache import MongoDBCache
from langchain_singlestore.cache import SingleStoreSemanticCache
from langchain_text_splitters import CharacterTextSplitter
from pymemcache.client.base import Client
from redis import Redis
from sqlalchemy import Column, Computed, Index, Integer, Sequence, String, create_engine
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import TSVectorType
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
from typing import Any, Dict
from typing import Any, Dict, List
from upstash_redis import Redis
from upstash_semantic_cache import SemanticCache
import cassio
import hashlib
import json
import langchain
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
# Model caches

This notebook covers how to cache results of individual LLM calls using different caches.

First, let's install some dependencies
"""
logger.info("# Model caches")

# %pip install -qU langchain-ollama langchain-community

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()


llm = Ollama(model="llama3.2", n=2, best_of=2)

"""
## `In Memory` cache
"""
logger.info("## `In Memory` cache")


set_llm_cache(InMemoryCache())

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## `SQLite` cache
"""
logger.info("## `SQLite` cache")

# !rm .langchain.db


set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## `Upstash Redis` caches

### Standard cache
Use [Upstash Redis](https://upstash.com) to cache prompts and responses with a serverless HTTP API.
"""
logger.info("## `Upstash Redis` caches")

# %pip install -qU upstash-redis


URL = "<UPSTASH_REDIS_REST_URL>"
TOKEN = "<UPSTASH_REDIS_REST_TOKEN>"

langchain.llm_cache = UpstashRedisCache(redis_=Redis(url=URL, token=TOKEN))

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
### Semantic cache

Use [Upstash Vector](https://upstash.com/docs/vector/overall/whatisvector) to do a semantic similarity search and cache the most similar response in the database. The vectorization is automatically done by the selected embedding model while creating Upstash Vector database.
"""
logger.info("### Semantic cache")

# %pip install upstash-semantic-cache


UPSTASH_VECTOR_REST_URL = "<UPSTASH_VECTOR_REST_URL>"
UPSTASH_VECTOR_REST_TOKEN = "<UPSTASH_VECTOR_REST_TOKEN>"

cache = SemanticCache(
    url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN, min_proximity=0.7
)

set_llm_cache(cache)

# %%time
llm.invoke("Which city is the most crowded city in the USA?")

# %%time
llm.invoke("Which city has the highest population in the USA?")

"""
## `Redis` caches

See the main [Redis cache docs](/docs/integrations/caches/redis_llm_caching/) for detail.

### Standard cache
Use [Redis](/docs/integrations/providers/redis) to cache prompts and responses.
"""
logger.info("## `Redis` caches")

# %pip install -qU redis


set_llm_cache(RedisCache(redis_=Redis()))

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
### Semantic cache
Use [Redis](/docs/integrations/providers/redis) to cache prompts and responses and evaluate hits based on semantic similarity.
"""
logger.info("### Semantic cache")

# %pip install -qU redis


set_llm_cache(
    RedisSemanticCache(redis_url="redis://localhost:6379", embedding=OllamaEmbeddings(model="mxbai-embed-large"))
)

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me one joke")

"""
## `GPTCache`

We can use [GPTCache](https://github.com/zilliztech/GPTCache) for exact match caching OR to cache results based on semantic similarity

Let's first start with an example of exact match
"""
logger.info("## `GPTCache`")

# %pip install -qU gptcache




def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


set_llm_cache(GPTCache(init_gptcache))

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
Let's now show an example of similarity caching
"""
logger.info("Let's now show an example of similarity caching")




def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}")


set_llm_cache(GPTCache(init_gptcache))

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me joke")

"""
## `MongoDB Atlas` caches

[MongoDB Atlas](https://www.mongodb.com/docs/atlas/) is a fully-managed cloud database available in AWS, Azure, and GCP. It has native support for 
Vector Search on the MongoDB document data.
Use [MongoDB Atlas Vector Search](/docs/integrations/providers/mongodb_atlas) to semantically cache prompts and responses.

### Standard cache

Standard cache is a simple cache in MongoDB. It does not use Semantic Caching, nor does it require an index to be made on the collection before generation.

To import this cache, first install the required dependency:

```bash
%pip install -qU langchain-mongodb
```

```python
```


To use this cache with your LLMs:
```python

# use any embedding provider...

mongodb_atlas_uri = "<YOUR_CONNECTION_STRING>"
COLLECTION_NAME="<YOUR_CACHE_COLLECTION_NAME>"
DATABASE_NAME="<YOUR_DATABASE_NAME>"

set_llm_cache(MongoDBCache(
    connection_string=mongodb_atlas_uri,
    collection_name=COLLECTION_NAME,
    database_name=DATABASE_NAME,
))
```


### Semantic cache

Semantic caching allows retrieval of cached prompts based on semantic similarity between the user input and previously cached results. Under the hood, it blends MongoDBAtlas as both a cache and a vectorstore.
The MongoDBAtlasSemanticCache inherits from `MongoDBAtlasVectorSearch` and needs an Atlas Vector Search Index defined to work. Please look at the [usage example](/docs/integrations/vectorstores/mongodb_atlas) on how to set up the index.

To import this cache:
```python
```

To use this cache with your LLMs:
```python

# use any embedding provider...

mongodb_atlas_uri = "<YOUR_CONNECTION_STRING>"
COLLECTION_NAME="<YOUR_CACHE_COLLECTION_NAME>"
DATABASE_NAME="<YOUR_DATABASE_NAME>"

set_llm_cache(MongoDBAtlasSemanticCache(
    embedding=FakeEmbeddings(),
    connection_string=mongodb_atlas_uri,
    collection_name=COLLECTION_NAME,
    database_name=DATABASE_NAME,
))
```

To find more resources about using MongoDBSemanticCache visit [here](https://www.mongodb.com/blog/post/introducing-semantic-caching-dedicated-mongodb-lang-chain-package-gen-ai-apps)

## `Momento` cache
Use [Momento](/docs/integrations/providers/momento) to cache prompts and responses.

Requires installing the `momento` package:
"""
logger.info("## `MongoDB Atlas` caches")

# %pip install -qU momento

"""
You'll need to get a Momento auth token to use this class. This can either be passed in to a momento.CacheClient if you'd like to instantiate that directly, as a named parameter `auth_token` to `MomentoChatMessageHistory.from_client_params`, or can just be set as an environment variable `MOMENTO_AUTH_TOKEN`.
"""
logger.info("You'll need to get a Momento auth token to use this class. This can either be passed in to a momento.CacheClient if you'd like to instantiate that directly, as a named parameter `auth_token` to `MomentoChatMessageHistory.from_client_params`, or can just be set as an environment variable `MOMENTO_AUTH_TOKEN`.")



cache_name = "langchain"
ttl = timedelta(days=1)
set_llm_cache(MomentoCache.from_client_params(cache_name, ttl))

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## `SQLAlchemy` cache

You can use `SQLAlchemyCache` to cache with any SQL database supported by `SQLAlchemy`.

### Standard cache
"""
logger.info("## `SQLAlchemy` cache")


engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
set_llm_cache(SQLAlchemyCache(engine))

"""
### Custom SQLAlchemy schemas

You can define your own declarative `SQLAlchemyCache` child class to customize the schema used for caching. For example, to support high-speed fulltext prompt indexing with `Postgres`, use:
"""
logger.info("### Custom SQLAlchemy schemas")


Base = declarative_base()


class FulltextLLMCache(Base):  # type: ignore
    """Postgres table for fulltext-indexed LLM Cache"""

    __tablename__ = "llm_cache_fulltext"
    id = Column(Integer, Sequence("cache_id"), primary_key=True)
    prompt = Column(String, nullable=False)
    llm = Column(String, nullable=False)
    idx = Column(Integer)
    response = Column(String)
    prompt_tsv = Column(
        TSVectorType(),
        Computed("to_tsvector('english', llm || ' ' || prompt)", persisted=True),
    )
    __table_args__ = (
        Index("idx_fulltext_prompt_tsv", prompt_tsv, postgresql_using="gin"),
    )


engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
set_llm_cache(SQLAlchemyCache(engine, FulltextLLMCache))

"""
## `Cassandra` caches

> [Apache Cassandra®](https://cassandra.apache.org/) is a NoSQL, row-oriented, highly scalable and highly available database. Starting with version 5.0, the database ships with [vector search capabilities](https://cassandra.apache.org/doc/trunk/cassandra/vector-search/overview.html).

You can use Cassandra for caching LLM responses, choosing from the exact-match `CassandraCache` or the (vector-similarity-based) `CassandraSemanticCache`.

Let's see both in action. The next cells guide you through the (little) required setup, and the following cells showcase the two available cache classes.

Required dependency:
"""
logger.info("## `Cassandra` caches")

# %pip install -qU "cassio>=0.1.4"

"""
### Connecting to the DB

The Cassandra caches shown in this page can be used with Cassandra as well as other derived databases that can use the CQL (Cassandra Query Language) protocol, such as DataStax Astra DB.

Depending on whether you connect to a Cassandra cluster or to Astra DB through CQL, you will provide different parameters when instantiating the cache (through initialization of a CassIO connection).

#### to a Cassandra cluster

You first need to create a `cassandra.cluster.Session` object, as described in the [Cassandra driver documentation](https://docs.datastax.com/en/developer/python-driver/latest/api/cassandra/cluster/#module-cassandra.cluster). The details vary (e.g. with network settings and authentication), but this might be something like:
"""
logger.info("### Connecting to the DB")


cluster = Cluster(["127.0.0.1"])
session = cluster.connect()

"""
You can now set the session, along with your desired keyspace name, as a global CassIO parameter:
"""
logger.info("You can now set the session, along with your desired keyspace name, as a global CassIO parameter:")


CASSANDRA_KEYSPACE = input("CASSANDRA_KEYSPACE = ")

cassio.init(session=session, keyspace=CASSANDRA_KEYSPACE)

"""
#### to Astra DB through CQL

In this case you initialize CassIO with the following connection parameters:

- the Database ID, e.g. `01234567-89ab-cdef-0123-456789abcdef`
- the Token, e.g. `AstraCS:6gBhNmsk135....` (it must be a "Database Administrator" token)
- Optionally a Keyspace name (if omitted, the default one for the database will be used)
"""
logger.info("#### to Astra DB through CQL")

# import getpass

ASTRA_DB_ID = input("ASTRA_DB_ID = ")
# ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")

desired_keyspace = input("ASTRA_DB_KEYSPACE (optional, can be left empty) = ")
if desired_keyspace:
    ASTRA_DB_KEYSPACE = desired_keyspace
else:
    ASTRA_DB_KEYSPACE = None


cassio.init(
    database_id=ASTRA_DB_ID,
    token=ASTRA_DB_APPLICATION_TOKEN,
    keyspace=ASTRA_DB_KEYSPACE,
)

"""
### Standard cache

This will avoid invoking the LLM when the supplied prompt is _exactly_ the same as one encountered already:
"""
logger.info("### Standard cache")


set_llm_cache(CassandraCache())

# %%time

logger.debug(llm.invoke("Why is the Moon always showing the same side?"))

# %%time

logger.debug(llm.invoke("Why is the Moon always showing the same side?"))

"""
### Semantic cache

This cache will do a semantic similarity search and return a hit if it finds a cached entry that is similar enough, For this, you need to provide an `Embeddings` instance of your choice.
"""
logger.info("### Semantic cache")


embedding = OllamaEmbeddings(model="mxbai-embed-large")


set_llm_cache(
    CassandraSemanticCache(
        embedding=embedding,
        table_name="my_semantic_cache",
    )
)

# %%time

logger.debug(llm.invoke("Why is the Moon always showing the same side?"))

# %%time

logger.debug(llm.invoke("How come we always see one face of the moon?"))

"""
**Attribution statement:**

>`Apache Cassandra`, `Cassandra` and `Apache` are either registered trademarks or trademarks of the [Apache Software Foundation](http://www.apache.org/) in the United States and/or other countries.

## `Astra DB` caches

You can easily use [Astra DB](https://docs.datastax.com/en/astra/home/astra.html) as an LLM cache, with either the "exact" or the "semantic-based" cache.

> [DataStax Astra DB](https://docs.datastax.com/en/astra-db-serverless/index.html) is a serverless 
> AI-ready database built on `Apache Cassandra®` and made conveniently available 
> through an easy-to-use JSON API.

_This approach differs from the `Cassandra` caches mentioned above in that it natively uses the HTTP Data API. The Data API is specific to Astra DB. Keep in mind that the storage format will also differ._

Make sure you have a running database (it must be a Vector-enabled database to use the Semantic cache) and get the required credentials on your Astra dashboard:

- the API Endpoint looks like `https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com`
- the Token looks like `AstraCS:6gBhNmsk135....`
"""
logger.info("## `Astra DB` caches")

# %pip install -qU langchain-astradb

# import getpass

ASTRA_DB_API_ENDPOINT = input("ASTRA_DB_API_ENDPOINT = ")
# ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")

"""
### Standard cache

This will avoid invoking the LLM when the supplied prompt is _exactly_ the same as one encountered already:
"""
logger.info("### Standard cache")


set_llm_cache(
    AstraDBCache(
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )
)

# %%time

logger.debug(llm.invoke("Is a true fakery the same as a fake truth?"))

# %%time

logger.debug(llm.invoke("Is a true fakery the same as a fake truth?"))

"""
### Semantic cache

This cache will do a semantic similarity search and return a hit if it finds a cached entry that is similar enough, For this, you need to provide an `Embeddings` instance of your choice.
"""
logger.info("### Semantic cache")


embedding = OllamaEmbeddings(model="mxbai-embed-large")


set_llm_cache(
    AstraDBSemanticCache(
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        embedding=embedding,
        collection_name="demo_semantic_cache",
    )
)

# %%time

logger.debug(llm.invoke("Are there truths that are false?"))

# %%time

logger.debug(llm.invoke("Is is possible that something false can be also true?"))

"""
## `Azure Cosmos DB` semantic cache

You can use this integrated [vector database](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database) for caching.
"""
logger.info("## `Azure Cosmos DB` semantic cache")



NAMESPACE = "langchain_test_db.langchain_test_collection"
CONNECTION_STRING = (
    "Please provide your azure cosmos mongo vCore vector db connection string"
)

DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

num_lists = 3
dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_IVF
m = 16
ef_construction = 64
ef_search = 40
score_threshold = 0.9
application_name = "LANGCHAIN_CACHING_PYTHON"


set_llm_cache(
    AzureCosmosDBSemanticCache(
        cosmosdb_connection_string=CONNECTION_STRING,
        cosmosdb_client=None,
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        database_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        num_lists=num_lists,
        similarity=similarity_algorithm,
        kind=kind,
        dimensions=dimensions,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        score_threshold=score_threshold,
        application_name=application_name,
    )
)

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## `Azure Cosmos DB NoSql` semantic cache

You can use this integrated [vector database](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database) for caching.
"""
logger.info("## `Azure Cosmos DB NoSql` semantic cache")



HOST = "COSMOS_DB_URI"
KEY = "COSMOS_DB_KEY"

cosmos_client = CosmosClient(HOST, KEY)


def get_vector_indexing_policy() -> dict:
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
    }


def get_vector_embedding_policy() -> dict:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "dimensions": 1536,
                "distanceFunction": "cosine",
            }
        ]
    }


cosmos_container_properties_test = {"partition_key": PartitionKey(path="/id")}
cosmos_database_properties_test: Dict[str, Any] = {}

set_llm_cache(
    AzureCosmosDBNoSqlSemanticCache(
        cosmos_client=cosmos_client,
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        vector_embedding_policy=get_vector_embedding_policy(),
        indexing_policy=get_vector_indexing_policy(),
        cosmos_container_properties=cosmos_container_properties_test,
        cosmos_database_properties=cosmos_database_properties_test,
    )
)

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## `Elasticsearch` caches

A caching layer for LLMs that uses Elasticsearch.

First install the LangChain integration with Elasticsearch.
"""
logger.info("## `Elasticsearch` caches")

# %pip install -qU langchain-elasticsearch

"""
### Standard cache

Use the class `ElasticsearchCache`.

Simple example:
"""
logger.info("### Standard cache")


set_llm_cache(
    ElasticsearchCache(
        es_url="http://localhost:9200",
        index_name="llm-chat-cache",
        metadata={"project": "my_chatgpt_project"},
    )
)

"""
The `index_name` parameter can also accept aliases. This allows to use the 
[ILM: Manage the index lifecycle](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html)
that we suggest to consider for managing retention and controlling cache growth.

Look at the class docstring for all parameters.

### Index the generated text

The cached data won't be searchable by default.
The developer can customize the building of the Elasticsearch document in order to add indexed text fields,
where to put, for example, the text generated by the LLM.

This can be done by subclassing end overriding methods.
The new cache class can be applied also to a pre-existing cache index:
"""
logger.info("### Index the generated text")




class SearchableElasticsearchCache(ElasticsearchCache):
    @property
    def mapping(self) -> Dict[str, Any]:
        mapping = super().mapping
        mapping["mappings"]["properties"]["parsed_llm_output"] = {
            "type": "text",
            "analyzer": "english",
        }
        return mapping

    def build_document(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> Dict[str, Any]:
        body = super().build_document(prompt, llm_string, return_val)
        body["parsed_llm_output"] = self._parse_output(body["llm_output"])
        return body

    @staticmethod
    def _parse_output(data: List[str]) -> List[str]:
        return [
            json.loads(output)["kwargs"]["message"]["kwargs"]["content"]
            for output in data
        ]


set_llm_cache(
    SearchableElasticsearchCache(
        es_url="http://localhost:9200", index_name="llm-chat-cache"
    )
)

"""
When overriding the mapping and the document building, 
please only make additive modifications, keeping the base mapping intact.

### Embedding cache

An Elasticsearch store for caching embeddings.
"""
logger.info("### Embedding cache")


"""
## LLM-specific optional caching

You can also turn off caching for specific LLMs. In the example below, even though global caching is enabled, we turn it off for a specific LLM.
"""
logger.info("## LLM-specific optional caching")

llm = Ollama(model="llama3.2", n=2, best_of=2, cache=False)

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## Optional caching in Chains

You can also turn off caching for particular nodes in chains. Note that because of certain interfaces, its often easier to construct the chain first, and then edit the LLM afterwards.

As an example, we will load a summarizer map-reduce chain. We will cache results for the map-step, but then not freeze it for the combine step.
"""
logger.info("## Optional caching in Chains")

llm = Ollama(model="llama3.2")
no_cache_llm = Ollama(model="llama3.2", cache=False)


text_splitter = CharacterTextSplitter()

with open("../how_to/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)


docs = [Document(page_content=t) for t in texts[:3]]

chain = load_summarize_chain(llm, chain_type="map_reduce", reduce_llm=no_cache_llm)

# %%time
chain.invoke(docs)

"""
When we run it again, we see that it runs substantially faster but the final answer is different. This is due to caching at the map steps, but not at the reduce step.
"""
logger.info("When we run it again, we see that it runs substantially faster but the final answer is different. This is due to caching at the map steps, but not at the reduce step.")

# %%time
chain.invoke(docs)

# !rm .langchain.db sqlite.db

"""


## `OpenSearch` semantic cache
Use [OpenSearch](https://python.langchain.com/docs/integrations/vectorstores/opensearch/) as a semantic cache to cache prompts and responses and evaluate hits based on semantic similarity.
"""
logger.info("## `OpenSearch` semantic cache")


set_llm_cache(
    OpenSearchSemanticCache(
        opensearch_url="http://localhost:9200", embedding=OllamaEmbeddings(model="mxbai-embed-large")
    )
)

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me one joke")

"""
## `SingleStoreDB` semantic cache

You can use [SingleStore](https://python.langchain.com/docs/integrations/vectorstores/singlestore/) as a semantic cache to cache prompts and responses.
"""
logger.info("## `SingleStoreDB` semantic cache")

# %pip install -qU langchain-singlestore


set_llm_cache(
    SingleStoreSemanticCache(
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        host="root:pass@localhost:3306/db",
    )
)

"""
## `Memcached` cache
You can use [Memcached](https://www.memcached.org/) as a cache to cache prompts and responses through [pymemcache](https://github.com/pinterest/pymemcache).

This cache requires the pymemcache dependency to be installed:
"""
logger.info("## `Memcached` cache")

# %pip install -qU pymemcache


set_llm_cache(MemcachedCache(Client("localhost")))

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
## `Couchbase` caches

Use [Couchbase](https://couchbase.com/) as a cache for prompts and responses.

### Standard cache

The standard cache that looks for an exact match of the user prompt.
"""
logger.info("## `Couchbase` caches")

# %pip install -qU langchain-couchbase



COUCHBASE_CONNECTION_STRING = (
    "couchbase://localhost"  # or "couchbases://localhost" if using TLS
)
DB_USERNAME = "Administrator"
DB_PASSWORD = "Password"

auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
options = ClusterOptions(auth)
cluster = Cluster(COUCHBASE_CONNECTION_STRING, options)

cluster.wait_until_ready(timedelta(seconds=5))

BUCKET_NAME = "langchain-testing"
SCOPE_NAME = "_default"
COLLECTION_NAME = "_default"

set_llm_cache(
    CouchbaseCache(
        cluster=cluster,
        bucket_name=BUCKET_NAME,
        scope_name=SCOPE_NAME,
        collection_name=COLLECTION_NAME,
    )
)

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me a joke")

"""
#### Time to Live (TTL) for the cached entries
The Cached documents can be deleted after a specified time automatically by specifying a `ttl` parameter along with the initialization of the Cache.
"""
logger.info("#### Time to Live (TTL) for the cached entries")


set_llm_cache(
    CouchbaseCache(
        cluster=cluster,
        bucket_name=BUCKET_NAME,
        scope_name=SCOPE_NAME,
        collection_name=COLLECTION_NAME,
        ttl=timedelta(minutes=5),
    )
)

"""
### Semantic cache
Semantic caching allows users to retrieve cached prompts based on semantic similarity between the user input and previously cached inputs. Under the hood it uses Couchbase as both a cache and a vectorstore. This needs an appropriate Vector Search Index defined to work. Please look at the usage example on how to set up the index.
"""
logger.info("### Semantic cache")



COUCHBASE_CONNECTION_STRING = (
    "couchbase://localhost"  # or "couchbases://localhost" if using TLS
)
DB_USERNAME = "Administrator"
DB_PASSWORD = "Password"

auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
options = ClusterOptions(auth)
cluster = Cluster(COUCHBASE_CONNECTION_STRING, options)

cluster.wait_until_ready(timedelta(seconds=5))

"""
Notes:
- The search index for the semantic cache needs to be defined before using the semantic cache. 
- The optional parameter, `score_threshold` in the Semantic Cache that you can use to tune the results of the semantic search.

#### Index to the Full Text Search service

How to Import an Index to the Full Text Search service?
 - [Couchbase Server](https://docs.couchbase.com/server/current/search/import-search-index.html)
     - Click on Search -> Add Index -> Import
     - Copy the following Index definition in the Import screen
     - Click on Create Index to create the index.
 - [Couchbase Capella](https://docs.couchbase.com/cloud/search/import-search-index.html)
     - Copy the index definition to a new file `index.json`
     - Import the file in Capella using the instructions in the documentation.
     - Click on Create Index to create the index.

**Example index for the vector search:**

  ```
  {
    "type": "fulltext-index",
    "name": "langchain-testing._default.semantic-cache-index",
    "sourceType": "gocbcore",
    "sourceName": "langchain-testing",
    "planParams": {
      "maxPartitionsPerPIndex": 1024,
      "indexPartitions": 16
    },
    "params": {
      "doc_config": {
        "docid_prefix_delim": "",
        "docid_regexp": "",
        "mode": "scope.collection.type_field",
        "type_field": "type"
      },
      "mapping": {
        "analysis": {},
        "default_analyzer": "standard",
        "default_datetime_parser": "dateTimeOptional",
        "default_field": "_all",
        "default_mapping": {
          "dynamic": true,
          "enabled": false
        },
        "default_type": "_default",
        "docvalues_dynamic": false,
        "index_dynamic": true,
        "store_dynamic": true,
        "type_field": "_type",
        "types": {
          "_default.semantic-cache": {
            "dynamic": false,
            "enabled": true,
            "properties": {
              "embedding": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "dims": 1536,
                    "index": true,
                    "name": "embedding",
                    "similarity": "dot_product",
                    "type": "vector",
                    "vector_index_optimized_for": "recall"
                  }
                ]
              },
              "metadata": {
                "dynamic": true,
                "enabled": true
              },
              "text": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "index": true,
                    "name": "text",
                    "store": true,
                    "type": "text"
                  }
                ]
              }
            }
          }
        }
      },
      "store": {
        "indexType": "scorch",
        "segmentVersion": 16
      }
    },
    "sourceParams": {}
  }
  ```
"""
logger.info("#### Index to the Full Text Search service")

BUCKET_NAME = "langchain-testing"
SCOPE_NAME = "_default"
COLLECTION_NAME = "semantic-cache"
INDEX_NAME = "semantic-cache-index"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

cache = CouchbaseSemanticCache(
    cluster=cluster,
    embedding=embeddings,
    bucket_name=BUCKET_NAME,
    scope_name=SCOPE_NAME,
    collection_name=COLLECTION_NAME,
    index_name=INDEX_NAME,
    score_threshold=0.8,
)

set_llm_cache(cache)

# %%time
logger.debug(llm.invoke("How long do dogs live?"))

# %%time
logger.debug(llm.invoke("What is the expected lifespan of a dog?"))

"""
#### Time to Live (TTL) for the cached entries

The Cached documents can be deleted after a specified time automatically by specifying a `ttl` parameter along with the initialization of the Cache.
"""
logger.info("#### Time to Live (TTL) for the cached entries")


set_llm_cache(
    CouchbaseSemanticCache(
        cluster=cluster,
        embedding=embeddings,
        bucket_name=BUCKET_NAME,
        scope_name=SCOPE_NAME,
        collection_name=COLLECTION_NAME,
        index_name=INDEX_NAME,
        score_threshold=0.8,
        ttl=timedelta(minutes=5),
    )
)

"""
## Cache classes: summary table

**Cache** classes are implemented by inheriting the [BaseCache](https://python.langchain.com/api_reference/core/caches/langchain_core.caches.BaseCache.html) class.

This table lists all derived classes with links to the API Reference.


| Namespace | Class 🔻 |
|------------|---------|
| langchain_astradb.cache | [AstraDBCache](https://python.langchain.com/api_reference/astradb/cache/langchain_astradb.cache.AstraDBCache.html) |
| langchain_astradb.cache | [AstraDBSemanticCache](https://python.langchain.com/api_reference/astradb/cache/langchain_astradb.cache.AstraDBSemanticCache.html) |
| langchain_community.cache | [AstraDBCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.AstraDBCache.html) (deprecated since `langchain-community==0.0.28`) |
| langchain_community.cache | [AstraDBSemanticCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.AstraDBSemanticCache.html) (deprecated since `langchain-community==0.0.28`) |
| langchain_community.cache | [AzureCosmosDBSemanticCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.AzureCosmosDBSemanticCache.html) |
| langchain_community.cache | [CassandraCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.CassandraCache.html) |
| langchain_community.cache | [CassandraSemanticCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.CassandraSemanticCache.html) |
| langchain_couchbase.cache | [CouchbaseCache](https://python.langchain.com/api_reference/couchbase/cache/langchain_couchbase.cache.CouchbaseCache.html) |
| langchain_couchbase.cache | [CouchbaseSemanticCache](https://python.langchain.com/api_reference/couchbase/cache/langchain_couchbase.cache.CouchbaseSemanticCache.html) |
| langchain_elasticsearch.cache | [ElasticsearchCache](https://python.langchain.com/api_reference/elasticsearch/cache/langchain_elasticsearch.cache.AsyncElasticsearchCache.html) |
| langchain_elasticsearch.cache | [ElasticsearchEmbeddingsCache](https://python.langchain.com/api_reference/elasticsearch/cache/langchain_elasticsearch.cache.AsyncElasticsearchEmbeddingsCache.html) |
| langchain_community.cache | [GPTCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.GPTCache.html) |
| langchain_core.caches | [InMemoryCache](https://python.langchain.com/api_reference/core/caches/langchain_core.caches.InMemoryCache.html) |
| langchain_community.cache | [InMemoryCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.InMemoryCache.html) |
| langchain_community.cache | [MomentoCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.MomentoCache.html) |
| langchain_mongodb.cache | [MongoDBAtlasSemanticCache](https://python.langchain.com/api_reference/mongodb/cache/langchain_mongodb.cache.MongoDBAtlasSemanticCache.html) |
| langchain_mongodb.cache | [MongoDBCache](https://python.langchain.com/api_reference/mongodb/cache/langchain_mongodb.cache.MongoDBCache.html) |
| langchain_community.cache | [OpenSearchSemanticCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.OpenSearchSemanticCache.html) |
| langchain_community.cache | [RedisSemanticCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.RedisSemanticCache.html) |
| langchain_community.cache | [SingleStoreDBSemanticCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.SingleStoreDBSemanticCache.html) |
| langchain_community.cache | [SQLAlchemyCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.SQLAlchemyCache.html) |
| langchain_community.cache | [SQLAlchemyMd5Cache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.SQLAlchemyMd5Cache.html) |
| langchain_community.cache | [UpstashRedisCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.UpstashRedisCache.html) |
"""
logger.info("## Cache classes: summary table")


logger.info("\n\n[DONE]", bright=True)