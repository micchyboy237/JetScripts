from jet.logger import logger
from langchain_community.retrievers import (
ElasticSearchBM25Retriever,
)
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
# ElasticSearch BM25

>[Elasticsearch](https://www.elastic.co/elasticsearch/) is a distributed, RESTful search and analytics engine. It provides a distributed, multitenant-capable full-text search engine with an HTTP web interface and schema-free JSON documents.

>In information retrieval, [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (BM is an abbreviation of best matching) is a ranking function used by search engines to estimate the relevance of documents to a given search query. It is based on the probabilistic retrieval framework developed in the 1970s and 1980s by Stephen E. Robertson, Karen Spärck Jones, and others.

>The name of the actual ranking function is BM25. The fuller name, Okapi BM25, includes the name of the first system to use it, which was the Okapi information retrieval system, implemented at London's City University in the 1980s and 1990s. BM25 and its newer variants, e.g. BM25F (a version of BM25 that can take document structure and anchor text into account), represent TF-IDF-like retrieval functions used in document retrieval.

This notebook shows how to use a retriever that uses `ElasticSearch` and `BM25`.

For more information on the details of BM25 see [this blog post](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables).
"""
logger.info("# ElasticSearch BM25")

# %pip install --upgrade --quiet  elasticsearch


"""
## Create New Retriever
"""
logger.info("## Create New Retriever")

elasticsearch_url = "http://localhost:9200"
retriever = ElasticSearchBM25Retriever.create(elasticsearch_url, "langchain-index-4")


"""
## Add texts (if necessary)

We can optionally add texts to the retriever (if they aren't already in there)
"""
logger.info("## Add texts (if necessary)")

retriever.add_texts(["foo", "bar", "world", "hello", "foo bar"])

"""
## Use Retriever

We can now use the retriever!
"""
logger.info("## Use Retriever")

result = retriever.invoke("foo")

result

logger.info("\n\n[DONE]", bright=True)