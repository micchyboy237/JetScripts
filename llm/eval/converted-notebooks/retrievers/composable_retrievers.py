from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.storage.docstore.dynamodb import DynamoDBDocumentStore
from llama_index.storage.docstore.firestore import FirestoreDocumentStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import TokenTextSplitter
import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import download_loader
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Composable Objects
#
# In this notebook, we show how you can combine multiple objects into a single top-level index.
#
# This approach works by setting up `IndexNode` objects, with an `obj` field that points to a:
# - query engine
# - retriever
# - query pipeline
# - another node!
#
# ```python
# object = IndexNode(index_id="my_object", obj=query_engine, text="some text about this object")
# ```

# Data Setup

# %pip install llama-index-storage-docstore-mongodb
# %pip install llama-index-vector-stores-qdrant
# %pip install llama-index-storage-docstore-firestore
# %pip install llama-index-retrievers-bm25
# %pip install llama-index-storage-docstore-redis
# %pip install llama-index-storage-docstore-dynamodb
# %pip install llama-index-readers-file pymupdf

# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "./llama2.pdf"
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/1706.03762.pdf" -O "./attention.pdf"


llama2_docs = PyMuPDFReader().load_data(
    file_path="./llama2.pdf", metadata=True
)
attention_docs = PyMuPDFReader().load_data(
    file_path="./attention.pdf", metadata=True
)

# Retriever Setup


# os.environ["OPENAI_API_KEY"] = "sk-..."


nodes = TokenTextSplitter(
    chunk_size=1024, chunk_overlap=128
).get_nodes_from_documents(llama2_docs + attention_docs)


docstore = SimpleDocumentStore()
docstore.add_documents(nodes)


client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore("composable", client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes=nodes)
vector_retriever = index.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore, similarity_top_k=2
)

# Composing Objects
#
# Here, we construct the `IndexNodes`. Note that the text is what is used to index the node by the top-level index.
#
# For a vector index, the text is embedded, for a keyword index, the text is used for keywords.
#
# In this example, the `SummaryIndex` is used, which does not technically need the text for retrieval, since it always retrieves all nodes.


vector_obj = IndexNode(
    index_id="vector", obj=vector_retriever, text="Vector Retriever"
)
bm25_obj = IndexNode(
    index_id="bm25", obj=bm25_retriever, text="BM25 Retriever"
)


summary_index = SummaryIndex(objects=[vector_obj, bm25_obj])

# Querying
#
# When we query, all objects will be retrieved and used to generate the nodes to get a final answer.
#
# Using `tree_summarize` with `aquery()` ensures concurrent execution and faster responses.

query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize", verbose=True
)

response = query_engine.query(
    "How does attention work in transformers?"
)

print(str(response))

response = query_engine.query(
    "What is the architecture of Llama2 based on?"
)

print(str(response))

response = query_engine.query(
    "What was used before attention in transformers?"
)

print(str(response))

# Note on Saving and Loading
#
# Since objects aren't technically serializable, when saving and loading, then need to be provided at load time as well.
#
# Here's an example of how I might save/load this setup.

# Save

docstore.persist("./docstore.json")

# Load


docstore = SimpleDocumentStore.from_persist_path("./docstore.json")

client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore("composable", client=client)

index = VectorStoreIndex.from_vector_store(vector_store)
vector_retriever = index.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore, similarity_top_k=2
)


vector_obj = IndexNode(
    index_id="vector", obj=vector_retriever, text="Vector Retriever"
)
bm25_obj = IndexNode(
    index_id="bm25", obj=bm25_retriever, text="BM25 Retriever"
)


summary_index = SummaryIndex(objects=[vector_obj, bm25_obj])

logger.info("\n\n[DONE]", bright=True)
