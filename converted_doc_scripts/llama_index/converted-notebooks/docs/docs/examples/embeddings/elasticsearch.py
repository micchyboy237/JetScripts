from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.elasticsearch import ElasticsearchEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/elasticsearch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Elasticsearch Embeddings

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Elasticsearch Embeddings")

# %pip install llama-index-vector-stores-elasticsearch
# %pip install llama-index-embeddings-elasticsearch

# !pip install llama-index



host = os.environ.get("ES_HOST", "localhost:9200")
username = os.environ.get("ES_USERNAME", "elastic")
password = os.environ.get("ES_PASSWORD", "changeme")
index_name = os.environ.get("INDEX_NAME", "your-index-name")
model_id = os.environ.get("MODEL_ID", "your-model-id")


embeddings = ElasticsearchEmbedding.from_credentials(
    model_id=model_id, es_url=host, es_username=username, es_password=password
)

Settings.embed_model = embeddings
Settings.chunk_size = 512

vector_store = ElasticsearchStore(
    index_name=index_name, es_url=host, es_user=username, es_password=password
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
)

query_engine = index.as_query_engine()


response = query_engine.query("hello world")

logger.info("\n\n[DONE]", bright=True)