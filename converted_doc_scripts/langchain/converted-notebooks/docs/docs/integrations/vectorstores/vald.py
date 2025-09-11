from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Vald
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import grpc
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
# Vald

> [Vald](https://github.com/vdaas/vald) is a highly scalable distributed fast approximate nearest neighbor (ANN) dense vector search engine.

This notebook shows how to use functionality related to the `Vald` database.

To run this notebook you need a running Vald cluster.
Check [Get Started](https://github.com/vdaas/vald#get-started) for more information.

See the [installation instructions](https://github.com/vdaas/vald-client-python#install).
"""
logger.info("# Vald")

# %pip install --upgrade --quiet  vald-client-python langchain-community

"""
## Basic Example
"""
logger.info("## Basic Example")


raw_documents = TextLoader("state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
db = Vald.from_documents(documents, embeddings, host="localhost", port=8080)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
docs[0].page_content

"""
### Similarity search by vector
"""
logger.info("### Similarity search by vector")

embedding_vector = embeddings.embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
docs[0].page_content

"""
### Similarity search with score
"""
logger.info("### Similarity search with score")

docs_and_scores = db.similarity_search_with_score(query)
docs_and_scores[0]

"""
## Maximal Marginal Relevance Search (MMR)

In addition to using similarity search in the retriever object, you can also use `mmr` as retriever.
"""
logger.info("## Maximal Marginal Relevance Search (MMR)")

retriever = db.as_retriever(search_type="mmr")
retriever.invoke(query)

"""
Or use `max_marginal_relevance_search` directly:
"""
logger.info("Or use `max_marginal_relevance_search` directly:")

db.max_marginal_relevance_search(query, k=2, fetch_k=10)

"""
## Example of using secure connection
In order to run this notebook, it is necessary to run a Vald cluster with secure connection.

Here is an example of a Vald cluster with the following configuration using [Athenz](https://github.com/AthenZ/athenz) authentication.

ingress(TLS) -> [authorization-proxy](https://github.com/AthenZ/authorization-proxy)(Check athenz-role-auth in grpc metadata) -> vald-lb-gateway
"""
logger.info("## Example of using secure connection")


with open("test_root_cacert.crt", "rb") as root:
    credentials = grpc.ssl_channel_credentials(root_certificates=root.read())

with open(".ztoken", "rb") as ztoken:
    token = ztoken.read().strip()

metadata = [(b"athenz-role-auth", token)]


raw_documents = TextLoader("state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

db = Vald.from_documents(
    documents,
    embeddings,
    host="localhost",
    port=443,
    grpc_use_secure=True,
    grpc_credentials=credentials,
    grpc_metadata=metadata,
)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query, grpc_metadata=metadata)
docs[0].page_content

"""
### Similarity search by vector
"""
logger.info("### Similarity search by vector")

embedding_vector = embeddings.embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector, grpc_metadata=metadata)
docs[0].page_content

"""
### Similarity search with score
"""
logger.info("### Similarity search with score")

docs_and_scores = db.similarity_search_with_score(query, grpc_metadata=metadata)
docs_and_scores[0]

"""
### Maximal Marginal Relevance Search (MMR)
"""
logger.info("### Maximal Marginal Relevance Search (MMR)")

retriever = db.as_retriever(
    search_kwargs={"search_type": "mmr", "grpc_metadata": metadata}
)
retriever.invoke(query, grpc_metadata=metadata)

"""
Or:
"""
logger.info("Or:")

db.max_marginal_relevance_search(query, k=2, fetch_k=10, grpc_metadata=metadata)

logger.info("\n\n[DONE]", bright=True)