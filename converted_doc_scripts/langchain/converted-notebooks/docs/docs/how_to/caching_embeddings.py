from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
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
# Caching

[Embeddings](/docs/concepts/embedding_models/) can be stored or temporarily cached to avoid needing to recompute them.

Caching embeddings can be done using a `CacheBackedEmbeddings`. The cache backed embedder is a wrapper around an embedder that caches
embeddings in a key-value store. The text is hashed and the hash is used as the key in the cache.

The main supported way to initialize a `CacheBackedEmbeddings` is `from_bytes_store`. It takes the following parameters:

- underlying_embedder: The embedder to use for embedding.
- document_embedding_cache: Any [`ByteStore`](/docs/integrations/stores/) for caching document embeddings.
- batch_size: (optional, defaults to `None`) The number of documents to embed between store updates.
- namespace: (optional, defaults to `""`) The namespace to use for document cache. This namespace is used to avoid collisions with other caches. For example, set it to the name of the embedding model used.
- query_embedding_cache: (optional, defaults to `None` or not caching) A [`ByteStore`](/docs/integrations/stores/) for caching query embeddings, or `True` to use the same store as `document_embedding_cache`.

**Attention**:

- Be sure to set the `namespace` parameter to avoid collisions of the same text embedded using different embeddings models.
- `CacheBackedEmbeddings` does not cache query embeddings by default. To enable query caching, one needs to specify a `query_embedding_cache`.
"""
logger.info("# Caching")


"""
## Using with a Vector Store

First, let's see an example that uses the local file system for storing embeddings and uses FAISS vector store for retrieval.
"""
logger.info("## Using with a Vector Store")

# %pip install --upgrade --quiet  langchain-ollama faiss-cpu


underlying_embeddings = OllamaEmbeddings(model="nomic-embed-text")

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)

"""
The cache is empty prior to embedding:
"""
logger.info("The cache is empty prior to embedding:")

list(store.yield_keys())

"""
Load the document, split it into chunks, embed each chunk and load it into the vector store.
"""
logger.info(
    "Load the document, split it into chunks, embed each chunk and load it into the vector store.")

raw_documents = TextLoader("state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

"""
Create the vector store:
"""
logger.info("Create the vector store:")

# %%time
db = FAISS.from_documents(documents, cached_embedder)

"""
If we try to create the vector store again, it'll be much faster since it does not need to re-compute any embeddings.
"""
logger.info("If we try to create the vector store again, it'll be much faster since it does not need to re-compute any embeddings.")

# %%time
db2 = FAISS.from_documents(documents, cached_embedder)

"""
And here are some of the embeddings that got created:
"""
logger.info("And here are some of the embeddings that got created:")

list(store.yield_keys())[:5]

"""
# Swapping the `ByteStore`

In order to use a different `ByteStore`, just use it when creating your `CacheBackedEmbeddings`. Below, we create an equivalent cached embeddings object, except using the non-persistent `InMemoryByteStore` instead:
"""
logger.info("# Swapping the `ByteStore`")


store = InMemoryByteStore()

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)

logger.info("\n\n[DONE]", bright=True)
