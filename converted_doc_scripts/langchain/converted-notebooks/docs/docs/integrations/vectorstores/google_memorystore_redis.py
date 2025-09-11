from google.colab import auth
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_google_memorystore_redis import (
DistanceStrategy,
HNSWConfig,
RedisVectorStore,
)
from langchain_text_splitters import CharacterTextSplitter
import os
import pprint
import redis
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
# Google Memorystore for Redis

> [Google Memorystore for Redis](https://cloud.google.com/memorystore/docs/redis/memorystore-for-redis-overview) is a fully-managed service that is powered by the Redis in-memory data store to build application caches that provide sub-millisecond data access. Extend your database application to build AI-powered experiences leveraging Memorystore for Redis's Langchain integrations.

This notebook goes over how to use [Memorystore for Redis](https://cloud.google.com/memorystore/docs/redis/memorystore-for-redis-overview) to store vector embeddings with the `MemorystoreVectorStore` class.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-memorystore-redis-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-memorystore-redis-python/blob/main/docs/vector_store.ipynb)

## Pre-reqs

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Memorystore for Redis API](https://console.cloud.google.com/flows/enableapi?apiid=redis.googleapis.com)
* [Create a Memorystore for Redis instance](https://cloud.google.com/memorystore/docs/redis/create-instance-console). Ensure that the version is greater than or equal to 7.2.

### 🦜🔗 Library Installation

The integration lives in its own `langchain-google-memorystore-redis` package, so we need to install it.
"""
logger.info("# Google Memorystore for Redis")

# %pip install -upgrade --quiet langchain-google-memorystore-redis langchain

"""
**Colab only:** Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.
"""



"""
### ☁ Set Your Google Cloud Project
Set your Google Cloud project so that you can leverage Google Cloud resources within this notebook.

If you don't know your project ID, try the following:

* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113).
"""
logger.info("### ☁ Set Your Google Cloud Project")

PROJECT_ID = "my-project-id"  # @param {type:"string"}

# !gcloud config set project {PROJECT_ID}

"""
### 🔐 Authentication
Authenticate to Google Cloud as the IAM user logged into this notebook in order to access your Google Cloud Project.

* If you are using Colab to run this notebook, use the cell below and continue.
* If you are using Vertex AI Workbench, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).
"""
logger.info("### 🔐 Authentication")


auth.authenticate_user()

"""
## Basic Usage

### Initialize a Vector Index
"""
logger.info("## Basic Usage")


redis_client = redis.from_url("redis://127.0.0.1:6379")

index_config = HNSWConfig(
    name="my_vector_index", distance_strategy=DistanceStrategy.COSINE, vector_size=128
)

RedisVectorStore.init_index(client=redis_client, index_config=index_config)

"""
### Prepare Documents

Text needs processing and numerical representation before interacting with a vector store. This involves:

* Loading Text: The TextLoader obtains text data from a file (e.g., "state_of_the_union.txt").
* Text Splitting: The CharacterTextSplitter breaks the text into smaller chunks for embedding models.
"""
logger.info("### Prepare Documents")


loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

"""
### Add Documents to the Vector Store

After text preparation and embedding generation, the following methods insert them into the Redis vector store.

#### Method 1: Classmethod for Direct Insertion

This approach combines embedding creation and insertion into a single step using the from_documents classmethod:
"""
logger.info("### Add Documents to the Vector Store")


embeddings = FakeEmbeddings(size=128)
redis_client = redis.from_url("redis://127.0.0.1:6379")
rvs = RedisVectorStore.from_documents(
    docs, embedding=embeddings, client=redis_client, index_name="my_vector_index"
)

"""
#### Method 2: Instance-Based Insertion
This approach offers flexibility when working with a new or existing RedisVectorStore:

* [Optional] Create a RedisVectorStore Instance: Instantiate a RedisVectorStore object for customization. If you already have an instance, proceed to the next step.
* Add Text with Metadata: Provide raw text and metadata to the instance. Embedding generation and insertion into the vector store are handled automatically.
"""
logger.info("#### Method 2: Instance-Based Insertion")

rvs = RedisVectorStore(
    client=redis_client, index_name="my_vector_index", embeddings=embeddings
)
ids = rvs.add_texts(
    texts=[d.page_content for d in docs], metadatas=[d.metadata for d in docs]
)

"""
### Perform a Similarity Search (KNN)

With the vector store populated, it's possible to search for text semantically similar to a query.  Here's how to use KNN (K-Nearest Neighbors) with default settings:

* Formulate the Query: A natural language question expresses the search intent (e.g., "What did the president say about Ketanji Brown Jackson").
* Retrieve Similar Results: The `similarity_search` method finds items in the vector store closest to the query in meaning.
"""
logger.info("### Perform a Similarity Search (KNN)")


query = "What did the president say about Ketanji Brown Jackson"
knn_results = rvs.similarity_search(query=query)
pprint.plogger.debug(knn_results)

"""
### Perform a Range-Based Similarity Search

Range queries provide more control by specifying a desired similarity threshold along with the query text:

* Formulate the Query: A natural language question defines the search intent.
* Set Similarity Threshold: The distance_threshold parameter determines how close a match must be considered relevant.
* Retrieve Results: The `similarity_search_with_score` method finds items from the vector store that fall within the specified similarity threshold.
"""
logger.info("### Perform a Range-Based Similarity Search")

rq_results = rvs.similarity_search_with_score(query=query, distance_threshold=0.8)
pprint.plogger.debug(rq_results)

"""
### Perform a Maximal Marginal Relevance (MMR) Search

MMR queries aim to find results that are both relevant to the query and diverse from each other, reducing redundancy in search results.

* Formulate the Query: A natural language question defines the search intent.
* Balance Relevance and Diversity: The lambda_mult parameter controls the trade-off between strict relevance and promoting variety in the results.
* Retrieve MMR Results: The `max_marginal_relevance_search` method returns items that optimize the combination of relevance and diversity based on the lambda setting.
"""
logger.info("### Perform a Maximal Marginal Relevance (MMR) Search")

mmr_results = rvs.max_marginal_relevance_search(query=query, lambda_mult=0.90)
pprint.plogger.debug(mmr_results)

"""
## Use the Vector Store as a Retriever

For seamless integration with other LangChain components, a vector store can be converted into a Retriever. This offers several advantages:

* LangChain Compatibility: Many LangChain tools and methods are designed to directly interact with retrievers.
* Ease of Use: The `as_retriever()` method converts the vector store into a format that simplifies querying.
"""
logger.info("## Use the Vector Store as a Retriever")

retriever = rvs.as_retriever()
results = retriever.invoke(query)
pprint.plogger.debug(results)

"""
## Clean up

### Delete Documents from the Vector Store

Occasionally, it's necessary to remove documents (and their associated vectors) from the vector store.  The `delete` method provides this functionality.
"""
logger.info("## Clean up")

rvs.delete(ids)

"""
### Delete a Vector Index

There might be circumstances where the deletion of an existing vector index is necessary. Common reasons include:

* Index Configuration Changes: If index parameters need modification, it's often required to delete and recreate the index.
* Storage Management: Removing unused indices can help free up space within the Redis instance.

Caution: Vector index deletion is an irreversible operation. Be certain that the stored vectors and search functionality are no longer required before proceeding.
"""
logger.info("### Delete a Vector Index")

RedisVectorStore.drop_index(client=redis_client, index_name="my_vector_index")

logger.info("\n\n[DONE]", bright=True)