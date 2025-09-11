from jet.models.config import MODELS_CACHE_DIR
from aerospike_vector_search import Client, HostPort
from aerospike_vector_search.types import VectorDistanceMetric
from jet.logger import logger
from langchain_aerospike.vectorstores import Aerospike
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import itertools
import os
import shutil
import tarfile


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
# Aerospike

[Aerospike Vector Search](https://aerospike.com/docs/vector) (AVS) is an
extension to the Aerospike Database that enables searches across very large
datasets stored in Aerospike. This new service lives outside of Aerospike and
builds an index to perform those searches.

This notebook showcases the functionality of the [LangChain Aerospike VectorStore
integration](https://github.com/aerospike/langchain-aerospike).

## Install AVS

Before using this notebook, we need to have a running AVS instance. Use one of
the [available installation methods](https://aerospike.com/docs/vector/install). 

When finished, store your AVS instance's IP address and port to use later
in this demo:
"""
logger.info("# Aerospike")

AVS_HOST = "<avs_ip>"
AVS_PORT = 5000

"""
## Install Dependencies 
The `sentence-transformers` dependency is large. This step could take several minutes to complete.
"""
logger.info("## Install Dependencies")

# !pip install --upgrade --quiet aerospike-vector-search==4.2.0 langchain-aerospike langchain-community sentence-transformers langchain

"""
## Download Quotes Dataset

We will download a dataset of approximately 100,000 quotes and use a subset of those quotes for semantic search.
"""
logger.info("## Download Quotes Dataset")

# !wget https://github.com/aerospike/aerospike-vector-search-examples/raw/7dfab0fccca0852a511c6803aba46578729694b5/quote-semantic-search/container-volumes/quote-search/data/quotes.csv.tgz

"""
## Load the Quotes Into Documents

We will load our quotes dataset using the `CSVLoader` document loader. In this case, `lazy_load` returns an iterator to ingest our quotes more efficiently. In this example, we only load 5,000 quotes.
"""
logger.info("## Load the Quotes Into Documents")



filename = "./quotes.csv"

if not os.path.exists(filename) and os.path.exists(filename + ".tgz"):
    with tarfile.open(filename + ".tgz", "r:gz") as tar:
        tar.extractall(path=os.path.dirname(filename))

NUM_QUOTES = 5000
documents = CSVLoader(filename, metadata_columns=["author", "category"]).lazy_load()
documents = list(
    itertools.islice(documents, NUM_QUOTES)
)  # Allows us to slice an iterator

logger.debug(documents[0])

"""
## Create your Embedder

In this step, we use HuggingFaceEmbeddings and the "all-MiniLM-L6-v2" sentence transformer model to embed our documents so we can perform a vector search.
"""
logger.info("## Create your Embedder")


MODEL_DIM = 384
MODEL_DISTANCE_CALC = VectorDistanceMetric.COSINE
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

"""
## Create an Aerospike Index and Embed Documents

Before we add documents, we need to create an index in the Aerospike Database. In the example below, we use some convenience code that checks to see if the expected index already exists.
"""
logger.info("## Create an Aerospike Index and Embed Documents")


seed = HostPort(host=AVS_HOST, port=AVS_PORT)

NAMESPACE = "test"

INDEX_NAME = "quote-miniLM-L6-v2"

VECTOR_KEY = "vector"

client = Client(seeds=seed)
index_exists = False

for index in client.index_list():
    if index["id"]["namespace"] == NAMESPACE and index["id"]["name"] == INDEX_NAME:
        index_exists = True
        logger.debug(f"{INDEX_NAME} already exists. Skipping creation")
        break

if not index_exists:
    logger.debug(f"{INDEX_NAME} does not exist. Creating index")
    client.index_create(
        namespace=NAMESPACE,
        name=INDEX_NAME,
        vector_field=VECTOR_KEY,
        vector_distance_metric=MODEL_DISTANCE_CALC,
        dimensions=MODEL_DIM,
        index_labels={
            "model": "miniLM-L6-v2",
            "date": "05/04/2024",
            "dim": str(MODEL_DIM),
            "distance": "cosine",
        },
    )

docstore = Aerospike.from_documents(
    documents,
    embedder,
    client=client,
    namespace=NAMESPACE,
    vector_key=VECTOR_KEY,
    index_name=INDEX_NAME,
    distance_strategy=MODEL_DISTANCE_CALC,
)

"""
## Search the Documents
Now that we have embedded our vectors, we can use vector search on our quotes.
"""
logger.info("## Search the Documents")

query = "A quote about the beauty of the cosmos"
docs = docstore.similarity_search(
    query, k=5, index_name=INDEX_NAME, metadata_keys=["_id", "author"]
)


def print_documents(docs):
    for i, doc in enumerate(docs):
        logger.debug("~~~~ Document", i, "~~~~")
        logger.debug("auto-generated id:", doc.metadata["_id"])
        logger.debug("author: ", doc.metadata["author"])
        logger.debug(doc.page_content)
        logger.debug("~~~~~~~~~~~~~~~~~~~~\n")


print_documents(docs)

"""
## Embedding Additional Quotes as Text

We can use `add_texts` to add additional quotes.
"""
logger.info("## Embedding Additional Quotes as Text")

docstore = Aerospike(
    client,
    embedder,
    NAMESPACE,
    index_name=INDEX_NAME,
    vector_key=VECTOR_KEY,
    distance_strategy=MODEL_DISTANCE_CALC,
)

ids = docstore.add_texts(
    [
        "quote: Rebellions are built on hope.",
        "quote: Logic is the beginning of wisdom, not the end.",
        "quote: If wishes were fishes, we’d all cast nets.",
    ],
    metadatas=[
        {"author": "Jyn Erso, Rogue One"},
        {"author": "Spock, Star Trek"},
        {"author": "Frank Herbert, Dune"},
    ],
)

logger.debug("New IDs")
logger.debug(ids)

"""
## Search Documents Using Max Marginal Relevance Search

We can use max marginal relevance search to find vectors that are similar to our query but dissimilar to each other. In this example, we create a retriever object using `as_retriever`, but this could be done just as easily by calling `docstore.max_marginal_relevance_search` directly. The `lambda_mult` search argument determines the diversity of our query response. 0 corresponds to maximum diversity and 1 to minimum diversity.
"""
logger.info("## Search Documents Using Max Marginal Relevance Search")

query = "A quote about our favorite four-legged pets"
retriever = docstore.as_retriever(
    search_type="mmr", search_kwargs={"fetch_k": 20, "lambda_mult": 0.7}
)
matched_docs = retriever.invoke(query)

print_documents(matched_docs)

"""
## Search Documents with a Relevance Threshold

Another useful feature is a similarity search with a relevance threshold. Generally, we only want results that are most similar to our query but also within some range of proximity. A relevance of 1 is most similar and a relevance of 0 is most dissimilar.
"""
logger.info("## Search Documents with a Relevance Threshold")

query = "A quote about stormy weather"
retriever = docstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.4
    },  # A greater value returns items with more relevance
)
matched_docs = retriever.invoke(query)

print_documents(matched_docs)

"""
## Clean up

We need to make sure we close our client to release resources and clean up threads.
"""
logger.info("## Clean up")

client.close()

"""
## Ready. Set. Search!

Now that you are up to speed with Aerospike Vector Search's LangChain integration, you have the power of the Aerospike Database and the LangChain ecosystem at your finger tips. Happy building!
"""
logger.info("## Ready. Set. Search!")

logger.info("\n\n[DONE]", bright=True)