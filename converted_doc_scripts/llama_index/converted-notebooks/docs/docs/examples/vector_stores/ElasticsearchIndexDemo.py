from jet.logger import CustomLogger
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.stores.elasticsearch import ElasticsearchStore
from llama_index.vector_stores.elasticsearch import AsyncBM25Strategy
from llama_index.vector_stores.elasticsearch import AsyncDenseVectorStrategy
from llama_index.vector_stores.elasticsearch import AsyncSparseVectorStrategy
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/ElasticsearchIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Elasticsearch Vector Store

Elasticsearch is a distributed, RESTful search and analytics engine built on top of Apache Lucene. It offers different retrieval options including dense vector retrieval, sparse vector retrieval, keyword search and hybrid search.

[Sign up](https://cloud.elastic.co/registration?utm_source=llama-index&utm_content=documentation) for a free trial of Elastic Cloud or run a local server like described below.

Requires Elasticsearch 8.9.0 or higher and AIOHTTP.
"""
logger.info("# Elasticsearch Vector Store")

# %pip install -qU llama-index-vector-stores-elasticsearch llama-index openai

# import getpass


# os.environ["OPENAI_API_KEY"] = getpass.getpass("OllamaFunctionCallingAdapter API Key:")

# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
## Running and connecting to Elasticsearch
Two ways to setup an Elasticsearch instance for use with:

### Elastic Cloud
Elastic Cloud is a managed Elasticsearch service. [Sign up](https://cloud.elastic.co/registration?utm_source=llama-index&utm_content=documentation) for a free trial.

### Locally
Get started with Elasticsearch by running it locally. The easiest way is to use the official Elasticsearch Docker image. See the Elasticsearch Docker documentation for more information.

```bash
docker run -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.license.self_generated.type=trial" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.2
```

## Configuring ElasticsearchStore
The ElasticsearchStore class is used to connect to an Elasticsearch instance. It requires the following parameters:

        - index_name: Name of the Elasticsearch index. Required.
        - es_client: Optional. Pre-existing Elasticsearch client.
        - es_url: Optional. Elasticsearch URL.
        - es_cloud_id: Optional. Elasticsearch cloud ID.
        - es_api_key: Optional. Elasticsearch API key.
        - es_user: Optional. Elasticsearch username.
        - es_password: Optional. Elasticsearch password.
        - text_field: Optional. Name of the Elasticsearch field that stores the text.
        - vector_field: Optional. Name of the Elasticsearch field that stores the
                    embedding.
        - batch_size: Optional. Batch size for bulk indexing. Defaults to 200.
        - distance_strategy: Optional. Distance strategy to use for similarity search.
                        Defaults to "COSINE".

### Example: Connecting locally
```python

es = ElasticsearchStore(
    index_name="my_index",
    es_url="http://localhost:9200",
)
```

### Example: Connecting to Elastic Cloud with username and password

```python

es = ElasticsearchStore(
    index_name="my_index",
    es_cloud_id="<cloud-id>", # found within the deployment page
    es_user="elastic"
    es_password="<password>" # provided when creating deployment. Alternatively can reset password.
)
```

### Example: Connecting to Elastic Cloud with API Key

```python

es = ElasticsearchStore(
    index_name="my_index",
    es_cloud_id="<cloud-id>", # found within the deployment page
    es_api_key="<api-key>" # create an API key within Kibana (Security -> API Keys)
)
```

#### Example data
"""
logger.info("## Running and connecting to Elasticsearch")


movies = [
    TextNode(
        text="The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        metadata={"title": "Pulp Fiction"},
    ),
    TextNode(
        text="When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
        metadata={"title": "The Dark Knight"},
    ),
    TextNode(
        text="An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
        metadata={"title": "Fight Club"},
    ),
    TextNode(
        text="A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into thed of a C.E.O.",
        metadata={"title": "Inception"},
    ),
    TextNode(
        text="A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        metadata={"title": "The Matrix"},
    ),
    TextNode(
        text="Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.",
        metadata={"title": "Se7en"},
    ),
    TextNode(
        text="An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.",
        metadata={"title": "The Godfather", "theme": "Mafia"},
    ),
]

"""
## Retrieval Examples

This section shows the different retrieval options available through the `ElasticsearchStore` and make use of them via a VectorStoreIndex.
"""
logger.info("## Retrieval Examples")


"""
We first define a helper function to retrieve and print results for user query input:
"""
logger.info("We first define a helper function to retrieve and print results for user query input:")

def print_results(results):
    for rank, result in enumerate(results, 1):
        logger.debug(
            f"{rank}. title={result.metadata['title']} score={result.get_score()} text={result.get_text()}"
        )


def search(
    vector_store: ElasticsearchStore, nodes: list[TextNode], query: str
):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    logger.debug(">>> Documents:")
    retriever = index.as_retriever()
    results = retriever.retrieve(query)
    print_results(results)

    logger.debug("\n>>> Answer:")
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    logger.debug(response)

"""
### Dense retrieval

Here we use embeddings from OllamaFunctionCallingAdapter to search.
"""
logger.info("### Dense retrieval")


dense_vector_store = ElasticsearchStore(
    es_url="http://localhost:9200",  # for Elastic Cloud authentication see above
    index_name="movies_dense",
    retrieval_strategy=AsyncDenseVectorStrategy(),
)

search(dense_vector_store, movies, "which movie involves dreaming?")

"""
This is also the default retrieval strategy:
"""
logger.info("This is also the default retrieval strategy:")

default_store = ElasticsearchStore(
    es_url="http://localhost:9200",  # for Elastic Cloud authentication see above
    index_name="movies_default",
)

search(default_store, movies, "which movie involves dreaming?")

"""
### Sparse retrieval

For this example you first need to [deploy the ELSER model](https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html) version two in your Elasticsearch deployment.
"""
logger.info("### Sparse retrieval")


sparse_vector_store = ElasticsearchStore(
    es_url="http://localhost:9200",  # for Elastic Cloud authentication see above
    index_name="movies_sparse",
    retrieval_strategy=AsyncSparseVectorStrategy(model_id=".elser_model_2"),
)

search(sparse_vector_store, movies, "which movie involves dreaming?")

"""
### Keyword retrieval

To use classic full-text search, you can use the BM25 strategy.
"""
logger.info("### Keyword retrieval")


bm25_store = ElasticsearchStore(
    es_url="http://localhost:9200",  # for Elastic Cloud authentication see above
    index_name="movies_bm25",
    retrieval_strategy=AsyncBM25Strategy(),
)

search(bm25_store, movies, "joker")

"""
### Hybrid retrieval

Combining dense retrieval and keyword search for hybrid retrieval can be enabled by setting a flag.
"""
logger.info("### Hybrid retrieval")


hybrid_store = ElasticsearchStore(
    es_url="http://localhost:9200",  # for Elastic Cloud authentication see above
    index_name="movies_hybrid",
    retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True),
)

search(hybrid_store, movies, "which movie involves dreaming?")

"""
### Metadata Filters

We can also apply filters to the query engine based on the metadata of our documents.
"""
logger.info("### Metadata Filters")


metadata_store = ElasticsearchStore(
    es_url="http://localhost:9200",  # for Elastic Cloud authentication see above
    index_name="movies_metadata",
)
storage_context = StorageContext.from_defaults(vector_store=metadata_store)
index = VectorStoreIndex(movies, storage_context=storage_context)

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)
retriever = index.as_retriever(filters=filters)

results = retriever.retrieve("What is inception about?")
print_results(results)

"""
## Custom Filters and overriding Query 
The elastic search implementation only supports ExactMatchFilters provided from LlamaIndex at the moment. Elasticsearch itself supports a wide range of filters, including range filters, geo filters, and more. To use these filters, you can pass them in as a list of dictionaries to the `es_filter` parameter.
"""
logger.info("## Custom Filters and overriding Query")

def custom_query(query, query_str):
    logger.debug("custom query", query)
    return query


query_engine = index.as_query_engine(
    vector_store_kwargs={
        "es_filter": [{"match": {"title": "matrix"}}],
        "custom_query": custom_query,
    }
)
query_engine.query("what is this movie about?")

logger.info("\n\n[DONE]", bright=True)