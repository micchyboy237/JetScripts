from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Rockset
from langchain_text_splitters import CharacterTextSplitter
import os
import rockset
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
# Rockset

>[Rockset](https://rockset.com/) is a real-time search and analytics database built for the cloud. Rockset uses a [Converged Index™](https://rockset.com/blog/converged-indexing-the-secret-sauce-behind-rocksets-fast-queries/) with an efficient store for vector embeddings to serve low latency, high concurrency search queries at scale. Rockset has full support for metadata filtering and  handles real-time ingestion for constantly updating, streaming data.

This notebook demonstrates how to use `Rockset` as a vector store in LangChain. Before getting started, make sure you have access to a `Rockset` account and an API key available. [Start your free trial today.](https://rockset.com/create/)

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

## Setting Up Your Environment

1. Leverage the `Rockset` console to create a [collection](https://rockset.com/docs/collections/) with the Write API as your source. In this walkthrough, we create a collection named `langchain_demo`. 
    
    Configure the following [ingest transformation](https://rockset.com/docs/ingest-transformation/) to mark your embeddings field and take advantage of performance and storage optimizations:


   (We used Ollama `text-embedding-ada-002` for this examples, where #length_of_vector_embedding = 1536)

```
SELECT _input.* EXCEPT(_meta), 
VECTOR_ENFORCE(_input.description_embedding, #length_of_vector_embedding, 'float') as description_embedding 
FROM _input
```

2. After creating your collection, use the console to retrieve an [API key](https://rockset.com/docs/iam/#users-api-keys-and-roles). For the purpose of this notebook, we assume you are using the `Oregon(us-west-2)` region.

3. Install the [rockset-python-client](https://github.com/rockset/rockset-python-client) to enable LangChain to communicate directly with `Rockset`.
"""
logger.info("# Rockset")

# %pip install --upgrade --quiet  rockset

"""
## LangChain Tutorial

Follow along in your own Python notebook to generate and store vector embeddings in Rockset.
Start using Rockset to search for documents similar to your search queries.

### 1. Define Key Variables
"""
logger.info("## LangChain Tutorial")



ROCKSET_API_KEY = os.environ.get(
    "ROCKSET_API_KEY"
)  # Verify ROCKSET_API_KEY environment variable
ROCKSET_API_SERVER = rockset.Regions.usw2a1  # Verify Rockset region
rockset_client = rockset.RocksetClient(ROCKSET_API_SERVER, ROCKSET_API_KEY)

COLLECTION_NAME = "langchain_demo"
TEXT_KEY = "description"
EMBEDDING_KEY = "description_embedding"

"""
### 2. Prepare Documents
"""
logger.info("### 2. Prepare Documents")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

"""
### 3. Insert Documents
"""
logger.info("### 3. Insert Documents")

# embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # Verify OPENAI_API_KEY environment variable

docsearch = Rockset(
    client=rockset_client,
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    text_key=TEXT_KEY,
    embedding_key=EMBEDDING_KEY,
)

ids = docsearch.add_texts(
    texts=[d.page_content for d in docs],
    metadatas=[d.metadata for d in docs],
)

"""
### 4. Search for Similar Documents
"""
logger.info("### 4. Search for Similar Documents")

query = "What did the president say about Ketanji Brown Jackson"
output = docsearch.similarity_search_with_relevance_scores(
    query, 4, Rockset.DistanceFunction.COSINE_SIM
)
logger.debug("output length:", len(output))
for d, dist in output:
    logger.debug(dist, d.metadata, d.page_content[:20] + "...")

"""
### 5. Search for Similar Documents with Filtering
"""
logger.info("### 5. Search for Similar Documents with Filtering")

output = docsearch.similarity_search_with_relevance_scores(
    query,
    4,
    Rockset.DistanceFunction.COSINE_SIM,
    where_str="{} NOT LIKE '%citizens%'".format(TEXT_KEY),
)
logger.debug("output length:", len(output))
for d, dist in output:
    logger.debug(dist, d.metadata, d.page_content[:20] + "...")

"""
### 6. [Optional] Delete Inserted Documents

You must have the unique ID associated with each document to delete them from your collection.
Define IDs when inserting documents with `Rockset.add_texts()`. Rockset will otherwise generate a unique ID for each document. Regardless, `Rockset.add_texts()` returns the IDs of inserted documents.

To delete these docs, simply use the `Rockset.delete_texts()` function.
"""
logger.info("### 6. [Optional] Delete Inserted Documents")

docsearch.delete_texts(ids)

"""
## Summary

In this tutorial, we successfully created a `Rockset` collection, `inserted` documents with  Ollama embeddings, and searched for similar documents with and without metadata filters.

Keep an eye on https://rockset.com/ for future updates in this space.
"""
logger.info("## Summary")

logger.info("\n\n[DONE]", bright=True)