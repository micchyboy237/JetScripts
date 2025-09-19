from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from datetime import timedelta
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## Couchbase Vector Store
[Couchbase](https://couchbase.com/) is an award-winning distributed NoSQL cloud database that delivers unmatched versatility, performance, scalability, and financial value for all of your cloud, mobile, AI, and edge computing applications. Couchbase embraces AI with coding assistance for developers and vector search for their applications.

Vector Search is a part of the [Full Text Search Service](https://docs.couchbase.com/server/current/learn/services-and-indexes/services/search-service.html) (Search Service) in Couchbase.

This tutorial explains how to use Vector Search in Couchbase. You can work with both [Couchbase Capella](https://www.couchbase.com/products/capella/) and your self-managed Couchbase Server.

### Installation
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("## Couchbase Vector Store")

# %pip install llama-index-vector-stores-couchbase

# !pip install llama-index

"""
### Creating Couchbase Connection
We create a connection to the Couchbase cluster initially and then pass the cluster object to the Vector Store.

Here, we are connecting using the username and password. You can also connect using any other supported way to your cluster.

For more information on connecting to the Couchbase cluster, please check the [Python SDK documentation](https://docs.couchbase.com/python-sdk/current/hello-world/start-using-sdk.html#connect).
"""
logger.info("### Creating Couchbase Connection")

COUCHBASE_CONNECTION_STRING = (
    "couchbase://localhost"  # or "couchbases://localhost" if using TLS
)
DB_USERNAME = "Administrator"
DB_PASSWORD = "P@ssword1!"



auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
options = ClusterOptions(auth)
cluster = Cluster(COUCHBASE_CONNECTION_STRING, options)

cluster.wait_until_ready(timedelta(seconds=5))

"""
### Creating the Search Index
Currently, the Search index needs to be created from the Couchbase Capella or Server UI or using the REST interface.

Let us define a Search index with the name `vector-index` on the `testing` bucket

For this example, let us use the Import Index feature on the Search Service on the UI.

We are defining an index on the testing bucket’s `_default` scope on the `_default` collection with the vector field set to `embedding` with 1536 dimensions and the text field set to text. We are also indexing and storing all the fields under metadata in the document as a dynamic mapping to account for varying document structures. The similarity metric is set to `dot_product`.

#### How to Import an Index to the Full Text Search service?

- [Couchbase Server](https://docs.couchbase.com/server/current/search/import-search-index.html)
    - Click on Search -> Add Index -> Import
    - Copy the following Index definition in the Import screen
    - Click on Create Index to create the index.


- [Couchbase Capella](https://docs.couchbase.com/cloud/search/import-search-index.html)
    - Copy the index definition to a new file `index.json`
    - Import the file in Capella using the instructions in the documentation.
    - Click on Create Index to create the index.

#### Index Definition
```
{
 "name": "vector-index",
 "type": "fulltext-index",
 "params": {
  "doc_config": {
   "docid_prefix_delim": "",
   "docid_regexp": "",
   "mode": "type_field",
   "type_field": "type"
  },
  "mapping": {
   "default_analyzer": "standard",
   "default_datetime_parser": "dateTimeOptional",
   "default_field": "_all",
   "default_mapping": {
    "dynamic": true,
    "enabled": true,
    "properties": {
     "metadata": {
      "dynamic": true,
      "enabled": true
     },
     "embedding": {
      "enabled": true,
      "dynamic": false,
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
     "text": {
      "enabled": true,
      "dynamic": false,
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
   },
   "default_type": "_default",
   "docvalues_dynamic": false,
   "index_dynamic": true,
   "store_dynamic": true,
   "type_field": "_type"
  },
  "store": {
   "indexType": "scorch",
   "segmentVersion": 16
  }
 },
 "sourceType": "gocbcore",
 "sourceName": "testing",
 "sourceParams": {},
 "planParams": {
  "maxPartitionsPerPIndex": 103,
  "indexPartitions": 10,
  "numReplicas": 0
 }
}
```

We will now set the bucket, scope, and collection names in the Couchbase cluster that we want to use for Vector Search.

For this example, we are using the default scope & collections.
"""
logger.info("### Creating the Search Index")

BUCKET_NAME = "testing"
SCOPE_NAME = "_default"
COLLECTION_NAME = "_default"
SEARCH_INDEX_NAME = "vector-index"


"""
For this tutorial, we will use OllamaFunctionCalling embeddings
"""
logger.info("For this tutorial, we will use OllamaFunctionCalling embeddings")

# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("OllamaFunctionCalling API Key:")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load the documents
"""
logger.info("#### Load the documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

vector_store = CouchbaseSearchVectorStore(
    cluster=cluster,
    bucket_name=BUCKET_NAME,
    scope_name=SCOPE_NAME,
    collection_name=COLLECTION_NAME,
    index_name=SEARCH_INDEX_NAME,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Basic Example
We will ask the query engine a question about the essay we just indexed.
"""
logger.info("### Basic Example")

query_engine = index.as_query_engine()
response = query_engine.query("What were his investments in Y Combinator?")
logger.debug(response)

"""
### Metadata Filters
We will create some example documents with metadata so that we can see how to filter documents based on metadata.
"""
logger.info("### Metadata Filters")


nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
        },
    ),
]
vector_store.add(nodes)


filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

retriever = index.as_retriever(filters=filters)

retriever.retrieve("What is inception about?")

"""
### Custom Filters and overriding Query
Couchbase supports `ExactMatchFilters` only at the moment via LlamaIndex. Couchbase supports a wide range of filters, including range filters, geospatial filters, and more. To use these filters, you can pass them in as a list of dictionaries to the `cb_search_options` parameter. 
The different search/query possibilities for the search_options can be found [here](https://docs.couchbase.com/server/current/search/search-request-params.html#query-object).
"""
logger.info("### Custom Filters and overriding Query")

def custom_query(query, query_str):
    logger.debug("custom query", query)
    return query


query_engine = index.as_query_engine(
    vector_store_kwargs={
        "cb_search_options": {
            "query": {"match": "growing up", "field": "text"}
        },
        "custom_query": custom_query,
    }
)
response = query_engine.query("what were his investments in Y Combinator?")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)