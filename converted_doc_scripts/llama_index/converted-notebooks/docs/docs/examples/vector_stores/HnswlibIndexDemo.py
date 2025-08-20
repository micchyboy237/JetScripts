from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
StorageContext,
SimpleDirectoryReader,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.hnswlib import HnswlibVectorStore
import hnswlib
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Hnswlib

Hnswlib is a fast approximate nearest neighbor search index. It's a lightweight, header-only C++ HNSW implementation that has no dependencies other than C++11. Hnswlib provides python bindings.
"""
logger.info("# Hnswlib")

# %pip install llama-index
# %pip install llama-index-vector-stores-hnswlib
# %pip install llama-index-embeddings-huggingface
# %pip install hnswlib

"""
### Import package dependencies
"""
logger.info("### Import package dependencies")


"""
### Load example data
"""
logger.info("### Load example data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Read the data
"""
logger.info("### Read the data")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug(f"Total documents: {len(documents)}")
logger.debug(f"First document, id: {documents[0].doc_id}")
logger.debug(f"First document, hash: {documents[0].hash}")
logger.debug(
    "First document, text"
    f" ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

"""
### Load the embedding model
"""
logger.info("### Load the embedding model")

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    normalize=True,
)

"""
### Create Hnswlib Vector Store object from Hnswlib.Index parameters
"""
logger.info("### Create Hnswlib Vector Store object from Hnswlib.Index parameters")

hnswlib_vector_store = HnswlibVectorStore.from_params(
    space="ip",
    dimension=embed_model._model.get_sentence_embedding_dimension(),
    max_elements=1000,
)

"""
Alternatively, You can create a Hnswlib.Index object Yourself.
"""
logger.info("Alternatively, You can create a Hnswlib.Index object Yourself.")


index = hnswlib.Index(
    "ip", embed_model._model.get_sentence_embedding_dimension()
)
index.init_index(max_elements=1000)

hnswlib_vector_store = HnswlibVectorStore(index)

"""
### Build index from documents
"""
logger.info("### Build index from documents")

hnswlib_storage_context = StorageContext.from_defaults(
    vector_store=hnswlib_vector_store
)
hnswlib_index = VectorStoreIndex.from_documents(
    documents,
    storage_context=hnswlib_storage_context,
    embed_model=embed_model,
    show_progress=True,
)

"""
### Query index
"""
logger.info("### Query index")

k = 5
query = "Before college I wrote what begginers should write."
hnswlib_vector_retriever = hnswlib_index.as_retriever(similarity_top_k=k)
nodes_with_scores = nodes_with_scores = hnswlib_vector_retriever.retrieve(
    query
)
for node in nodes_with_scores:
    logger.debug(f"Node {node.id_} | Score: {node.score:.3f} - {node.text[:120]}...")

logger.info("\n\n[DONE]", bright=True)