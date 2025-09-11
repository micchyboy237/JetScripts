from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.vectorstores.elastic_vector_search import ElasticKNNSearch
from langchain_community.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain_core.documents import Document
from langchain_elasticsearch import BM25Strategy
from langchain_elasticsearch import DenseVectorStrategy
from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch import ElasticsearchStore, DenseVectorScriptScoreStrategy
from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy
from langchain_elasticsearch import SparseVectorStrategy
from typing import Dict
from uuid import uuid4
import EmbeddingTabs from "@theme/EmbeddingTabs"
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
# Elasticsearch

>[Elasticsearch](https://www.elastic.co/elasticsearch/) is a distributed, RESTful search and analytics engine, capable of performing both vector and lexical search. It is built on top of the Apache Lucene library. 

This notebook shows how to use functionality related to the `Elasticsearch` vector store.

## Setup

In order to use the `Elasticsearch` vector search you must install the `langchain-elasticsearch` package.
"""
logger.info("# Elasticsearch")

# %pip install -qU langchain-elasticsearch

"""
### Credentials

There are two main ways to setup an Elasticsearch instance for use with:

1. Elastic Cloud: Elastic Cloud is a managed Elasticsearch service. Signup for a [free trial](https://cloud.elastic.co/registration?utm_source=langchain&utm_content=documentation).

To connect to an Elasticsearch instance that does not require
login credentials (starting the docker instance with security enabled), pass the Elasticsearch URL and index name along with the
embedding object to the constructor.

2. Local Install Elasticsearch: Get started with Elasticsearch by running it locally. The easiest way is to use the official Elasticsearch Docker image. See the [Elasticsearch Docker documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) for more information.


### Running Elasticsearch via Docker 
Example: Run a single-node Elasticsearch instance with security disabled. This is not recommended for production use.
"""
logger.info("### Credentials")

# %docker run -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.12.1

"""
### Running with Authentication
For production, we recommend you run with security enabled. To connect with login credentials, you can use the parameters `es_api_key` or `es_user` and `es_password`.


<EmbeddingTabs/>
"""
logger.info("### Running with Authentication")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")


elastic_vector_search = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="langchain_index",
    embedding=embeddings,
    es_user="elastic",
    es_password="changeme",
)

"""
#### How to obtain a password for the default "elastic" user?

To obtain your Elastic Cloud password for the default "elastic" user:
1. Log in to the Elastic Cloud console at https://cloud.elastic.co
2. Go to "Security" > "Users"
3. Locate the "elastic" user and click "Edit"
4. Click "Reset password"
5. Follow the prompts to reset the password

#### How to obtain an API key?

To obtain an API key:
1. Log in to the Elastic Cloud console at https://cloud.elastic.co
2. Open Kibana and go to Stack Management > API Keys
3. Click "Create API key"
4. Enter a name for the API key and click "Create"
5. Copy the API key and paste it into the `api_key` parameter

### Elastic Cloud

To connect to an Elasticsearch instance on Elastic Cloud, you can use either the `es_cloud_id` parameter or `es_url`.
"""
logger.info("#### How to obtain a password for the default "elastic" user?")

elastic_vector_search = ElasticsearchStore(
    es_cloud_id="<cloud_id>",
    index_name="test_index",
    embedding=embeddings,
    es_user="elastic",
    es_password="changeme",
)

"""
If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info(
    "If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")


"""
## Initialization

Elasticsearch is running locally on localhost:9200 with [docker](#running-elasticsearch-via-docker). For more details on how to connect to Elasticsearch from Elastic Cloud, see [connecting with authentication](#running-with-authentication) above.
"""
logger.info("## Initialization")


vector_store = ElasticsearchStore(
    "langchain-demo", embedding=embeddings, es_url="http://localhost:9201"
)

"""
## Manage vector store

### Add items to vector store
"""
logger.info("## Manage vector store")


document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)

"""
### Delete items from vector store
"""
logger.info("### Delete items from vector store")

vector_store.delete(ids=[uuids[-1]])

"""
## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. These examples also show how to use filtering when searching.

### Query directly

#### Similarity search

Performing a simple similarity search with filtering on metadata can be done as follows:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(
    query="LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter=[{"term": {"metadata.source.keyword": "tweet"}}],
)
for res in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

"""
#### Similarity search with score

If you want to execute a similarity search and receive the corresponding scores you can run:
"""
logger.info("#### Similarity search with score")

results = vector_store.similarity_search_with_score(
    query="Will it be hot tomorrow",
    k=1,
    filter=[{"term": {"metadata.source.keyword": "news"}}],
)
for doc, score in results:
    logger.debug(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

"""
### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.
"""
logger.info("### Query by turning into retriever")

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2}
)
retriever.invoke("Stealing from the bank is a crime")

"""
## Distance Similarity Algorithm

Elasticsearch supports the following vector distance similarity algorithms:

- cosine
- euclidean
- dot_product

The cosine similarity algorithm is the default.

You can specify the similarity Algorithm needed via the similarity parameter.

**NOTE**: Depending on the retrieval strategy, the similarity algorithm cannot be changed at query time. It is needed to be set when creating the index mapping for field. If you need to change the similarity algorithm, you need to delete the index and recreate it with the correct distance_strategy.
"""
logger.info("## Distance Similarity Algorithm")

db = ElasticsearchStore.from_documents(
    docs,
    embeddings,
    es_url="http://localhost:9200",
    index_name="test",
    distance_strategy="COSINE",
)

"""
## Retrieval Strategies

Elasticsearch has big advantages over other vector only databases from its ability to support a wide range of retrieval strategies. In this notebook we will configure `ElasticsearchStore` to support some of the most common retrieval strategies.

By default, `ElasticsearchStore` uses the `DenseVectorStrategy` (was called `ApproxRetrievalStrategy` prior to version 0.2.0).

### DenseVectorStrategy

This will return the top k most similar vectors to the query vector. The `k` parameter is set when the `ElasticsearchStore` is initialized. The default value is 10.
"""
logger.info("## Retrieval Strategies")


db = ElasticsearchStore.from_documents(
    docs,
    embeddings,
    es_url="http://localhost:9200",
    index_name="test",
    strategy=DenseVectorStrategy(),
)

docs = db.similarity_search(
    query="What did the president say about Ketanji Brown Jackson?", k=10
)

"""
#### Example: Hybrid retrieval with dense vector and keyword search

This example will show how to configure ElasticsearchStore to perform a hybrid retrieval, using a combination of approximate semantic search and keyword based search.

We use RRF to balance the two scores from different retrieval methods.

To enable hybrid retrieval, we need to set `hybrid=True` in the `DenseVectorStrategy` constructor.
"""
logger.info("#### Example: Hybrid retrieval with dense vector and keyword search")

db = ElasticsearchStore.from_documents(
    docs,
    embeddings,
    es_url="http://localhost:9200",
    index_name="test",
    strategy=DenseVectorStrategy(hybrid=True),
)

"""
When hybrid is enabled, the query performed will be a combination of approximate semantic search and keyword based search.

It will use rrf (Reciprocal Rank Fusion) to balance the two scores from different retrieval methods.

**Note**: RRF requires Elasticsearch 8.9.0 or above.
"""
logger.info("When hybrid is enabled, the query performed will be a combination of approximate semantic search and keyword based search.")

{
    "retriever": {
        "rrf": {
            "retrievers": [
                {
                    "standard": {
                        "query": {
                            "bool": {
                                "filter": [],
                                "must": [{"match": {"text": {"query": "foo"}}}],
                            }
                        },
                    },
                },
                {
                    "knn": {
                        "field": "vector",
                        "filter": [],
                        "k": 1,
                        "num_candidates": 50,
                        "query_vector": [1.0, ..., 0.0],
                    },
                },
            ]
        }
    }
}

"""
#### Example: Dense vector search with Embedding Model in Elasticsearch

This example will show how to configure `ElasticsearchStore` to use the embedding model deployed in Elasticsearch for dense vector retrieval.

To use this, specify the model_id in `DenseVectorStrategy` constructor via the `query_model_id` argument.

**NOTE**: This requires the model to be deployed and running in Elasticsearch ML node. See [notebook example](https://github.com/elastic/elasticsearch-labs/blob/main/notebooks/integrations/hugging-face/loading-model-from-hugging-face.ipynb) on how to deploy the model with `eland`.
"""
logger.info(
    "#### Example: Dense vector search with Embedding Model in Elasticsearch")

DENSE_SELF_DEPLOYED_INDEX_NAME = "test-dense-self-deployed"

db = ElasticsearchStore(
    es_cloud_id="<your cloud id>",
    es_user="elastic",
    es_password="<your password>",
    index_name=DENSE_SELF_DEPLOYED_INDEX_NAME,
    query_field="text_field",
    vector_query_field="vector_query_field.predicted_value",
    strategy=DenseVectorStrategy(
        model_id="sentence-transformers__all-minilm-l6-v2"),
)

db.client.ingest.put_pipeline(
    id="test_pipeline",
    processors=[
        {
            "inference": {
                "model_id": "sentence-transformers__all-minilm-l6-v2",
                "field_map": {"query_field": "text_field"},
                "target_field": "vector_query_field",
            }
        }
    ],
)

db.client.indices.create(
    index=DENSE_SELF_DEPLOYED_INDEX_NAME,
    mappings={
        "properties": {
            "text_field": {"type": "text"},
            "vector_query_field": {
                "properties": {
                    "predicted_value": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "l2_norm",
                    }
                }
            },
        }
    },
    settings={"index": {"default_pipeline": "test_pipeline"}},
)

db.from_texts(
    ["hello world"],
    es_cloud_id="<cloud id>",
    es_user="elastic",
    es_password="<cloud password>",
    index_name=DENSE_SELF_DEPLOYED_INDEX_NAME,
    query_field="text_field",
    vector_query_field="vector_query_field.predicted_value",
    strategy=DenseVectorStrategy(
        model_id="sentence-transformers__all-minilm-l6-v2"),
)

db.similarity_search("hello world", k=10)

"""
### SparseVectorStrategy (ELSER)

This strategy uses Elasticsearch's sparse vector retrieval to retrieve the top-k results. We only support our own "ELSER" embedding model for now.

**NOTE**: This requires the ELSER model to be deployed and running in Elasticsearch ml node.

To use this, specify `SparseVectorStrategy` (was called `SparseVectorRetrievalStrategy` prior to version 0.2.0) in the `ElasticsearchStore` constructor. You will need to provide a model ID.
"""
logger.info("### SparseVectorStrategy (ELSER)")


db = ElasticsearchStore.from_documents(
    docs,
    es_cloud_id="<cloud id>",
    es_user="elastic",
    es_password="<cloud password>",
    index_name="test-elser",
    strategy=SparseVectorStrategy(model_id=".elser_model_2"),
)

db.client.indices.refresh(index="test-elser")

results = db.similarity_search(
    "What did the president say about Ketanji Brown Jackson", k=4
)
logger.debug(results[0])

"""
### DenseVectorScriptScoreStrategy

This strategy uses Elasticsearch's script score query to perform exact vector retrieval (also known as brute force) to retrieve the top-k results. (This strategy was called `ExactRetrievalStrategy` prior to version 0.2.0.)

To use this, specify `DenseVectorScriptScoreStrategy` in `ElasticsearchStore` constructor.
"""
logger.info("### DenseVectorScriptScoreStrategy")


db = ElasticsearchStore.from_documents(
    docs,
    embeddings,
    es_url="http://localhost:9200",
    index_name="test",
    strategy=DenseVectorScriptScoreStrategy(),
)

"""
### BM25Strategy

Finally, you can use full-text keyword search.

To use this, specify `BM25Strategy` in `ElasticsearchStore` constructor.
"""
logger.info("### BM25Strategy")


db = ElasticsearchStore.from_documents(
    docs,
    es_url="http://localhost:9200",
    index_name="test",
    strategy=BM25Strategy(),
)

"""
### BM25RetrievalStrategy

This strategy allows the user to perform searches using pure BM25 without vector search.

To use this, specify `BM25RetrievalStrategy` in `ElasticsearchStore` constructor.

Note that in the example below, the embedding option is not specified, indicating that the search is conducted without using embeddings.
"""
logger.info("### BM25RetrievalStrategy")


db = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="test_index",
    strategy=ElasticsearchStore.BM25RetrievalStrategy(),
)

db.add_texts(
    ["foo", "foo bar", "foo bar baz", "bar", "bar baz", "baz"],
)

results = db.similarity_search(query="foo", k=10)
logger.debug(results)

"""
## Customise the Query

With `custom_query` parameter at search, you are able to adjust the query that is used to retrieve documents from Elasticsearch. This is useful if you want to use a more complex query, to support linear boosting of fields.
"""
logger.info("## Customise the Query")


def custom_query(query_body: dict, query: str):
    """Custom query to be used in Elasticsearch.
    Args:
        query_body (dict): Elasticsearch query body.
        query (str): Query string.
    Returns:
        dict: Elasticsearch query body.
    """
    logger.debug("Query Retriever created by the retrieval strategy:")
    logger.debug(query_body)
    logger.debug()

    new_query_body = {"query": {"match": {"text": query}}}

    logger.debug("Query thats actually used in Elasticsearch:")
    logger.debug(new_query_body)
    logger.debug()

    return new_query_body


results = db.similarity_search(
    "What did the president say about Ketanji Brown Jackson",
    k=4,
    custom_query=custom_query,
)
logger.debug("Results:")
logger.debug(results[0])

"""
## Customize the Document Builder

With `doc_builder` parameter at search, you are able to adjust how a Document is being built using data retrieved from Elasticsearch. This is especially useful if you have indices which were not created using Langchain.
"""
logger.info("## Customize the Document Builder")


def custom_document_builder(hit: Dict) -> Document:
    src = hit.get("_source", {})
    return Document(
        page_content=src.get("content", "Missing content!"),
        metadata={
            "page_number": src.get("page_number", -1),
            "original_filename": src.get("original_filename", "Missing filename!"),
        },
    )


results = db.similarity_search(
    "What did the president say about Ketanji Brown Jackson",
    k=4,
    doc_builder=custom_document_builder,
)
logger.debug("Results:")
logger.debug(results[0])

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/rag)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

# FAQ

## Question: Im getting timeout errors when indexing documents into Elasticsearch. How do I fix this?
One possible issue is your documents might take longer to index into Elasticsearch. ElasticsearchStore uses the Elasticsearch bulk API which has a few defaults that you can adjust to reduce the chance of timeout errors.

This is also a good idea when you're using SparseVectorRetrievalStrategy.

The defaults are:
- `chunk_size`: 500
- `max_chunk_bytes`: 100MB

To adjust these, you can pass in the `chunk_size` and `max_chunk_bytes` parameters to the ElasticsearchStore `add_texts` method.

```python
    vector_store.add_texts(
        texts,
        bulk_kwargs={
            "chunk_size": 50,
            "max_chunk_bytes": 200000000
        }
    )
```

# Upgrading to ElasticsearchStore

If you're already using Elasticsearch in your langchain based project, you may be using the old implementations: `ElasticVectorSearch` and `ElasticKNNSearch` which are now deprecated. We've introduced a new implementation called `ElasticsearchStore` which is more flexible and easier to use. This notebook will guide you through the process of upgrading to the new implementation.

## What's new?

The new implementation is now one class called `ElasticsearchStore` which can be used for approximate dense vector, exact dense vector, sparse vector (ELSER), BM25 retrieval and hybrid retrieval, via strategies.

## I am using ElasticKNNSearch

Old implementation:

```python


db = ElasticKNNSearch(
  elasticsearch_url="http://localhost:9200",
  index_name="test_index",
  embedding=embedding
)

```

New implementation:

```python


db = ElasticsearchStore(
  es_url="http://localhost:9200",
  index_name="test_index",
  embedding=embedding,
  # if you use the model_id
  # strategy=DenseVectorStrategy(model_id="test_model")
  # if you use hybrid search
  # strategy=DenseVectorStrategy(hybrid=True)
)

```

## I am using ElasticVectorSearch

Old implementation:

```python


db = ElasticVectorSearch(
  elasticsearch_url="http://localhost:9200",
  index_name="test_index",
  embedding=embedding
)

```

New implementation:

```python


db = ElasticsearchStore(
  es_url="http://localhost:9200",
  index_name="test_index",
  embedding=embedding,
  strategy=DenseVectorScriptScoreStrategy()
)

```

```python
db.client.indices.delete(
    index="test-metadata, test-elser, test-basic",
    ignore_unavailable=True,
    allow_no_indices=True,
)
```

## API reference

For detailed documentation of all `ElasticSearchStore` features and configurations head to the API reference: https://python.langchain.com/api_reference/elasticsearch/vectorstores/langchain_elasticsearch.vectorstores.ElasticsearchStore.html
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)
