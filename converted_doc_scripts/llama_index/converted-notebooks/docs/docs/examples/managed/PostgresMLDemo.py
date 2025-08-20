from google.colab import userdata
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.indices.managed.postgresml import PostgresMLIndex
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/managed/vectaraDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# PostgresML Managed Index
In this notebook we are going to show how to use [PostgresML](https://postgresml.org) with LlamaIndex.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# PostgresML Managed Index")

# !pip install llama-index-indices-managed-postgresml

# !pip install llama-index



# import nest_asyncio

# nest_asyncio.apply()

"""
### Loading documents
Load the `paul_graham_essay.txt` document.
"""
logger.info("### Loading documents")

# !mkdir data
# !curl -o data/paul_graham_essay.txt https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt

documents = SimpleDirectoryReader("data").load_data()
logger.debug(f"documents loaded into {len(documents)} document objects")
logger.debug(f"Document ID of first doc is {documents[0].doc_id}")

"""
### Upsert the documents into your PostgresML database

First let's set the url to our PostgresML database. If you don't have a url yet, you can make one for free here: https://postgresml.org/signup
"""
logger.info("### Upsert the documents into your PostgresML database")


PGML_DATABASE_URL = userdata.get("PGML_DATABASE_URL")

index = PostgresMLIndex.from_documents(
    documents,
    collection_name="llama-index-example-demo",
    pgml_database_url=PGML_DATABASE_URL,
)

"""
### Query the Postgresml Index
We can now ask questions using the PostgresMLIndex retriever.
"""
logger.info("### Query the Postgresml Index")

query = "What did the author write about?"

"""
We can use a retriever to list search our documents:
"""
logger.info("We can use a retriever to list search our documents:")

retriever = index.as_retriever()
response = retriever.retrieve(query)
texts = [t.node.text for t in response]

logger.debug("The Nodes:")
logger.debug(response)
logger.debug("\nThe Texts")
logger.debug(texts)

"""
PostgresML allows for easy re-reranking in the same query as doing retrieval:
"""
logger.info("PostgresML allows for easy re-reranking in the same query as doing retrieval:")

retriever = index.as_retriever(
    limit=2,  # Limit to returning the 2 most related Nodes
    rerank={
        "model": "mixedbread-ai/mxbai-rerank-base-v1",  # Use the mxbai-rerank-base model for reranking
        "num_documents_to_rerank": 100,  # Rerank up to 100 results returned from the vector search
    },
)
response = retriever.retrieve(query)
texts = [t.node.text for t in response]

logger.debug("The Nodes:")
logger.debug(response)
logger.debug("\nThe Texts")
logger.debug(texts)

"""
with the as_query_engine(), we can ask questions and get the response in one query:
"""
logger.info("with the as_query_engine(), we can ask questions and get the response in one query:")

query_engine = index.as_query_engine()
response = query_engine.query(query)

logger.debug("The Response:")
logger.debug(response)
logger.debug("\nThe Source Nodes:")
logger.debug(response.get_formatted_sources())

"""
Note that the "response" object above includes both the summary text but also the source documents used to provide this response (citations). Notice the source nodes are all from the same document. That is because we only uploaded one document which PostgresML automatically split before embedding for us. All parameters can be controlled. See the documentation for more information.

We can enable streaming by passing `streaming=True` when we create our query_engine.

**NOTE: Streaming is painfully slow on google collab due to their internet connectivity.**
"""
logger.info("Note that the "response" object above includes both the summary text but also the source documents used to provide this response (citations). Notice the source nodes are all from the same document. That is because we only uploaded one document which PostgresML automatically split before embedding for us. All parameters can be controlled. See the documentation for more information.")

query_engine = index.as_query_engine(streaming=True)
results = query_engine.query(query)
for text in results.response_gen:
    logger.debug(text, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)