from couchbase import search
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from datetime import timedelta
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_text_splitters import CharacterTextSplitter
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
# Couchbase
[Couchbase](http://couchbase.com/) is an award-winning distributed NoSQL cloud database that delivers unmatched versatility, performance, scalability, and financial value for all of your cloud, mobile, AI, and edge computing applications. Couchbase embraces AI with coding assistance for developers and vector search for their applications.

Vector Search is a part of the [Full Text Search Service](https://docs.couchbase.com/server/current/learn/services-and-indexes/services/search-service.html) (Search Service) in Couchbase.

This tutorial explains how to use Vector Search in Couchbase. You can work with either [Couchbase Capella](https://www.couchbase.com/products/capella/) and your self-managed Couchbase Server.

## Setup

To access the `CouchbaseSearchVectorStore` you first need to install the `langchain-couchbase` partner package:
"""
logger.info("# Couchbase")

pip install - qU langchain-couchbase

"""
### Credentials

Head over to the Couchbase [website](https://cloud.couchbase.com) and create a new connection, making sure to save your database username and password:
"""
logger.info("### Credentials")

# import getpass

# COUCHBASE_CONNECTION_STRING = getpass.getpass(
"Enter the connection string for the Couchbase cluster: "
)
    # DB_USERNAME = getpass.getpass("Enter the username for the Couchbase cluster: ")
    # DB_PASSWORD = getpass.getpass("Enter the password for the Couchbase cluster: ")

    """
If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
    logger.info(
  "If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



   """
## Initialization

Before instantiating we need to create a connection.

### Create Couchbase Connection Object

We create a connection to the Couchbase cluster initially and then pass the cluster object to the Vector Store. 

Here, we are connecting using the username and password from above. You can also connect using any other supported way to your cluster. 

For more information on connecting to the Couchbase cluster, please check the [documentation](https://docs.couchbase.com/python-sdk/current/hello-world/start-using-sdk.html#connect).
"""
   logger.info("## Initialization")



   auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
   options = ClusterOptions(auth)
   cluster = Cluster(COUCHBASE_CONNECTION_STRING, options)

    cluster.wait_until_ready(timedelta(seconds=5))

   """
We will now set the bucket, scope, and collection names in the Couchbase cluster that we want to use for Vector Search. 

For this example, we are using the default scope & collections.
"""
    logger.info(
     "We will now set the bucket, scope, and collection names in the Couchbase cluster that we want to use for Vector Search.")

      BUCKET_NAME = "langchain_bucket"
      SCOPE_NAME = "_default"
      COLLECTION_NAME = "_default"
      SEARCH_INDEX_NAME = "langchain-test-index"

      """
For details on how to create a Search index with support for Vector fields, please refer to the documentation.

- [Couchbase Capella](https://docs.couchbase.com/cloud/vector-search/create-vector-search-index-ui.html)
  
- [Couchbase Server](https://docs.couchbase.com/server/current/vector-search/create-vector-search-index-ui.html)

### Simple Instantiation

Below, we create the vector store object with the cluster information and the search index name. 


<EmbeddingTabs/>
"""
      logger.info("### Simple Instantiation")


        embeddings = OllamaEmbeddings(model="nomic-embed-text")


        vector_store = CouchbaseSearchVectorStore(
   cluster = cluster,
   bucket_name = BUCKET_NAME,
   scope_name = SCOPE_NAME,
   collection_name = COLLECTION_NAME,
   embedding = embeddings,
   index_name = SEARCH_INDEX_NAME,
)

    """
### Specify the Text & Embeddings Field

You can optionally specify the text & embeddings field for the document using the `text_key` and `embedding_key` fields.
"""
    logger.info("### Specify the Text & Embeddings Field")

    vector_store_specific = CouchbaseSearchVectorStore(
cluster = cluster,
bucket_name = BUCKET_NAME,
scope_name = SCOPE_NAME,
 collection_name = COLLECTION_NAME,
  embedding = embeddings,
   index_name = SEARCH_INDEX_NAME,
    text_key = "text",
    embedding_key = "embedding",
)

    """
## Manage vector store

Once you have created your vector store, we can interact with it by adding and deleting different items.

### Add items to vector store

We can add items to our vector store by using the `add_documents` function.
"""
    logger.info("## Manage vector store")



    document_1 = Document(
page_content = "I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
metadata = {"source": "tweet"},
)

    document_2 = Document(
page_content = "The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
metadata = {"source": "news"},
)

    document_3 = Document(
page_content = "Building an exciting new project with LangChain - come check it out!",
metadata = {"source": "tweet"},
)

    document_4 = Document(
page_content = "Robbers broke into the city bank and stole $1 million in cash.",
metadata = {"source": "news"},
)

    document_5 = Document(
page_content = "Wow! That was an amazing movie. I can't wait to see it again.",
metadata = {"source": "tweet"},
)

    document_6 = Document(
page_content = "Is the new iPhone worth the price? Read this review to find out.",
metadata = {"source": "website"},
)

    document_7 = Document(
page_content = "The top 10 soccer players in the world right now.",
metadata = {"source": "website"},
)

    document_8 = Document(
page_content = "LangGraph is the best framework for building stateful, agentic applications!",
metadata = {"source": "tweet"},
)

    document_9 = Document(
page_content = "The stock market is down 500 points today due to fears of a recession.",
metadata = {"source": "news"},
)

    document_10 = Document(
page_content = "I have a bad feeling I am going to get deleted :(",
metadata = {"source": "tweet"},
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

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

### Query directly

#### Similarity search

Performing a simple similarity search can be done as follows:
"""
    logger.info("## Query vector store")

    results = vector_store.similarity_search(
"LangChain provides abstractions to make working with LLMs easy",
k = 2,
)
    for res in results:
    logger.debug(f"* {res.page_content} [{res.metadata}]")

    """
#### Similarity search with Score

You can also fetch the scores for the results by calling the `similarity_search_with_score` method.
"""
    logger.info("#### Similarity search with Score")

    results = vector_store.similarity_search_with_score("Will it be hot tomorrow?", k=1)
    for res, score in results:
    logger.debug(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    """
### Filtering Results

You can filter the search results by specifying any filter on the text or metadata in the document that is supported by the Couchbase Search service. 

The `filter` can be any valid [SearchQuery](https://docs.couchbase.com/python-sdk/current/howtos/full-text-searching-with-sdk.html#search-queries) supported by the Couchbase Python SDK. These filters are applied before the Vector Search is performed. 

If you want to filter on one of the fields in the metadata, you need to specify it using `.`

For example, to fetch the `source` field in the metadata, you need to specify `metadata.source`.

Note that the filter needs to be supported by the Search Index.
"""
    logger.info("### Filtering Results")


    query = "Are there any concerning financial news?"
    filter_on_source = search.MatchQuery("news", field="metadata.source")
    results = vector_store.similarity_search_with_score(
   query, fields = ["metadata.source"], filter = filter_on_source, k = 5
)
    for res, score in results:
    logger.debug(f"* {res.page_content} [{res.metadata}] {score}")

    """
### Specifying Fields to Return

You can specify the fields to return from the document using `fields` parameter in the searches. These fields are returned as part of the `metadata` object in the returned Document. You can fetch any field that is stored in the Search index. The `text_key` of the document is returned as part of the document's `page_content`.

If you do not specify any fields to be fetched, all the fields stored in the index are returned.

If you want to fetch one of the fields in the metadata, you need to specify it using `.`

For example, to fetch the `source` field in the metadata, you need to specify `metadata.source`.
"""
    logger.info("### Specifying Fields to Return")

    query = "What did I eat for breakfast today?"
    results = vector_store.similarity_search(query, fields=["metadata.source"])
    logger.debug(results[0])

    """
### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains. 

Here is how to transform your vector store into a retriever and then invoke the retreiever with a simple query and filter.
"""
    logger.info("### Query by turning into retriever")

    retriever = vector_store.as_retriever(
search_type = "similarity",
search_kwargs = {"k": 1, "score_threshold": 0.5},
)
    filter_on_source = search.MatchQuery("news", field="metadata.source")
    retriever.invoke("Stealing from the bank is a crime",
                  filter = filter_on_source)

                   """
### Hybrid Queries

Couchbase allows you to do hybrid searches by combining Vector Search results with searches on non-vector fields of the document like the `metadata` object. 

The results will be based on the combination of the results from both Vector Search and the searches supported by Search Service. The scores of each of the component searches are added up to get the total score of the result.

To perform hybrid searches, there is an optional parameter, `search_options` that can be passed to all the similarity searches.  
The different search/query possibilities for the `search_options` can be found [here](https://docs.couchbase.com/server/current/search/search-request-params.html#query-object).

#### Create Diverse Metadata for Hybrid Search
In order to simulate hybrid search, let us create some random metadata from the existing documents. 
We uniformly add three fields to the metadata, `date` between 2010 & 2020, `rating` between 1 & 5 and `author` set to either John Doe or Jane Doe.
"""
                   logger.info("### Hybrid Queries")


                    loader= TextLoader("../../how_to/state_of_the_union.txt")
                    documents= loader.load()
                    text_splitter= CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                    docs= text_splitter.split_documents(documents)

                    for i, doc in enumerate(docs):
                    doc.metadata["date"]= f"{range(2010, 2020)[i % 10]}-01-01"
                    doc.metadata["rating"]= range(1, 6)[i % 5]
                    doc.metadata["author"]= ["John Doe", "Jane Doe"][i % 2]

                    vector_store.add_documents(docs)

                    query= "What did the president say about Ketanji Brown Jackson"
                    results= vector_store.similarity_search(query)
                    logger.debug(results[0].metadata)

                   """
### Query by Exact Value
We can search for exact matches on a textual field like the author in the `metadata` object.
"""
                    logger.info("### Query by Exact Value")

                    query= "What did the president say about Ketanji Brown Jackson"
                    results= vector_store.similarity_search(
  query,
  search_options = {
      "query": {"field": "metadata.author", "match": "John Doe"}},
  fields = ["metadata.author"],
)
    logger.debug(results[0])

    """
### Query by Partial Match
We can search for partial matches by specifying a fuzziness for the search. This is useful when you want to search for slight variations or misspellings of a search query.

Here, "Jae" is close (fuzziness of 1) to "Jane".
"""
    logger.info("### Query by Partial Match")

    query = "What did the president say about Ketanji Brown Jackson"
    results = vector_store.similarity_search(
query,
search_options = {
    "query": {"field": "metadata.author", "match": "Jae", "fuzziness": 1}
    },
fields = ["metadata.author"],
)
    logger.debug(results[0])

    """
### Query by Date Range Query
We can search for documents that are within a date range query on a date field like `metadata.date`.
"""
    logger.info("### Query by Date Range Query")

    query = "Any mention about independence?"
    results = vector_store.similarity_search(
query,
search_options = {
    "query": {
         "start": "2016-12-31",
          "end": "2017-01-02",
         "inclusive_start": True,
            "inclusive_end": False,
            "field": "metadata.date",
         }
    },
)
    logger.debug(results[0])

    """
### Query by Numeric Range Query
We can search for documents that are within a range for a numeric field like `metadata.rating`.
"""
    logger.info("### Query by Numeric Range Query")

    query = "Any mention about independence?"
    results = vector_store.similarity_search_with_score(
query,
search_options = {
    "query": {
         "min": 3,
          "max": 5,
         "inclusive_min": True,
            "inclusive_max": True,
            "field": "metadata.rating",
         }
    },
)
    logger.debug(results[0])

    """
### Combining Multiple Search Queries
Different search queries can be combined using AND (conjuncts) or OR (disjuncts) operators.

In this example, we are checking for documents with a rating between 3 & 4 and dated between 2015 & 2018.
"""
    logger.info("### Combining Multiple Search Queries")

    query = "Any mention about independence?"
    results = vector_store.similarity_search_with_score(
query,
search_options = {
    "query": {
         "conjuncts": [
              {"min": 3, "max": 4, "inclusive_max": True,
                "field": "metadata.rating"},
              {"start": "2016-12-31", "end": "2017-01-02",
                "field": "metadata.date"},
              ]
         }
    },
)
    logger.debug(results[0])

    """
**Note** 

The hybrid search results might contain documents that do not satisfy all the search parameters. This is due to the way the [scoring is calculated](https://docs.couchbase.com/server/current/search/run-searches.html#scoring). 
The score is a sum of both the vector search score and the queries in the hybrid search. If the Vector Search score is high, the combined score will be more than the results that match all the queries in the hybrid search. 
To avoid such results, please use the `filter` parameter instead of hybrid search.

### Combining Hybrid Search Query with Filters
Hybrid Search can be combined with filters to get the best of both hybrid search and the filters for results matching the requirements.

In this example, we are checking for documents with a rating between 3 & 5 and matching the string "independence" in the text field.
"""
    logger.info("### Combining Hybrid Search Query with Filters")

    filter_text = search.MatchQuery("independence", field="text")

    query = "Any mention about independence?"
    results = vector_store.similarity_search_with_score(
query,
search_options = {
    "query": {
         "min": 3,
          "max": 5,
         "inclusive_min": True,
            "inclusive_max": True,
            "field": "metadata.rating",
         }
    },
filter = filter_text,
)

    logger.debug(results[0])

    """
### Other Queries
Similarly, you can use any of the supported Query methods like Geo Distance, Polygon Search, Wildcard, Regular Expressions, etc in the `search_options` parameter. Please refer to the documentation for more details on the available query methods and their syntax.

- [Couchbase Capella](https://docs.couchbase.com/cloud/search/search-request-params.html#query-object)
- [Couchbase Server](https://docs.couchbase.com/server/current/search/search-request-params.html#query-object)

## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/rag)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

## Frequently Asked Questions

### Question: Should I create the Search index before creating the CouchbaseSearchVectorStore object?
Yes, currently you need to create the Search index before creating the `CouchbaseSearchVectoreStore` object.

### Question: I am not seeing all the fields that I specified in my search results. 

In Couchbase, we can only return the fields stored in the Search index. Please ensure that the field that you are trying to access in the search results is part of the Search index.

One way to handle this is to index and store a document's fields dynamically in the index. 

- In Capella, you need to go to "Advanced Mode" then under the chevron "General Settings" you can check "[X] Store Dynamic Fields" or "[X] Index Dynamic Fields"
- In Couchbase Server, in the Index Editor (not Quick Editor) under the chevron  "Advanced" you can check "[X] Store Dynamic Fields" or "[X] Index Dynamic Fields"

Note that these options will increase the size of the index.

For more details on dynamic mappings, please refer to the [documentation](https://docs.couchbase.com/cloud/search/customize-index.html).

### Question: I am unable to see the metadata object in my search results. 
This is most likely due to the `metadata` field in the document not being indexed and/or stored by the Couchbase Search index. In order to index the `metadata` field in the document, you need to add it to the index as a child mapping. 

If you select to map all the fields in the mapping, you will be able to search by all metadata fields. Alternatively, to optimize the index, you can select the specific fields inside `metadata` object to be indexed. You can refer to the [docs](https://docs.couchbase.com/cloud/search/customize-index.html) to learn more about indexing child mappings.

Creating Child Mappings

* [Couchbase Capella](https://docs.couchbase.com/cloud/search/create-child-mapping.html)
* [Couchbase Server](https://docs.couchbase.com/server/current/search/create-child-mapping.html)

### Question: What is the difference between filter and search_options / hybrid queries? 
Filters are [pre-filters](https://docs.couchbase.com/server/current/vector-search/pre-filtering-vector-search.html#about-pre-filtering) that are used to restrict the documents searched in a Search index. It is available in Couchbase Server 7.6.4 & higher.

Hybrid Queries are additional search queries that can be used to tune the results being returned from the search index. 

Both filters and hybrid search queries have the same capabilites with slightly different syntax. Filters are [SearchQuery](https://docs.couchbase.com/python-sdk/current/howtos/full-text-searching-with-sdk.html#search-queries) objects while the hybrid search queries are [dictionaries](https://docs.couchbase.com/server/current/search/search-request-params.html).

## API reference

For detailed documentation of all `CouchbaseSearchVectorStore` features and configurations head to the [API reference](https://couchbase-ecosystem.github.io/langchain-couchbase/langchain_couchbase.html#module-langchain_couchbase.vectorstores.search_vector_store)
"""
    logger.info("### Other Queries")

    logger.info("\n\n[DONE]", bright=True)
