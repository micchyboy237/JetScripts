from IPython.display import Markdown, display
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
StorageContext,
)
from llama_index.core import QueryBundle
from llama_index.core import Settings
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.typesense import TypesenseVectorStore
from typesense import Client
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/TypesenseDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Typesense Vector Store

#### Download Data
"""
logger.info("# Typesense Vector Store")

# %pip install llama-index-embeddings-ollama
# %pip install llama-index-vector-stores-typesense

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load documents, build the VectorStoreIndex
"""
logger.info("#### Load documents, build the VectorStoreIndex")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


typesense_client = Client(
    {
        "api_key": "xyz",
        "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
        "connection_timeout_seconds": 2,
    }
)
typesense_vector_store = TypesenseVectorStore(typesense_client)
storage_context = StorageContext.from_defaults(
    vector_store=typesense_vector_store
)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
#### Query Index
"""
logger.info("#### Query Index")


query_str = "What did the author do growing up?"
embed_model = MLXEmbedding()

query_embedding = embed_model.get_agg_embedding_from_queries(query_str)
query_bundle = QueryBundle(query_str, embedding=query_embedding)
response = index.as_query_engine().query(query_bundle)

display(Markdown(f"<b>{response}</b>"))



query_bundle = QueryBundle(query_str=query_str)
response = index.as_query_engine(
    vector_store_query_mode=VectorStoreQueryMode.TEXT_SEARCH
).query(query_bundle)
display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)