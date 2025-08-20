from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import logging
import openai
import os
import shutil
import sys
import weaviate


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Weaviate Vector Store - Hybrid Search

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Weaviate Vector Store - Hybrid Search")

# %pip install llama-index-vector-stores-weaviate

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
## Creating a Weaviate Client
"""
logger.info("## Creating a Weaviate Client")


# os.environ["OPENAI_API_KEY"] = ""
# openai.api_key = os.environ["OPENAI_API_KEY"]


cluster_url = ""
api_key = ""

client = weaviate.connect_to_wcs(
    cluster_url=cluster_url,
    auth_credentials=weaviate.auth.AuthApiKey(api_key),
)


"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load documents
"""
logger.info("## Load documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Build the VectorStoreIndex with WeaviateVectorStore
"""
logger.info("## Build the VectorStoreIndex with WeaviateVectorStore")



vector_store = WeaviateVectorStore(weaviate_client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
## Query Index with Default Vector Search
"""
logger.info("## Query Index with Default Vector Search")

query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("What did the author do growing up?")

display_response(response)

"""
## Query Index with Hybrid Search

Use hybrid search with bm25 and vector.  
`alpha` parameter determines weighting (alpha = 0 -> bm25, alpha=1 -> vector search).

### By default, `alpha=0.75` is used (very similar to vector search)
"""
logger.info("## Query Index with Hybrid Search")

query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid", similarity_top_k=2
)
response = query_engine.query(
    "What did the author do growing up?",
)

display_response(response)

"""
### Set `alpha=0.` to favor bm25
"""
logger.info("### Set `alpha=0.` to favor bm25")

query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid", similarity_top_k=2, alpha=0.0
)
response = query_engine.query(
    "What did the author do growing up?",
)

display_response(response)

logger.info("\n\n[DONE]", bright=True)