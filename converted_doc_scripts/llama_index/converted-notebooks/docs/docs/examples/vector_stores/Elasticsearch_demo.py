from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
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

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/Elasticsearch_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Elasticsearch

>[Elasticsearch](http://www.github.com/elastic/elasticsearch) is a search database, that supports full text and vector searches.

## Basic Example

In this basic example, we take the a Paul Graham essay, split it into chunks, embed it using an open-source embedding model, load it into Elasticsearch, and then query it. For an example using different retrieval strategies see [Elasticsearch Vector Store](https://docs.llamaindex.ai/en/stable/examples/vector_stores/ElasticsearchIndexDemo/).

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Elasticsearch")

# %pip install -qU llama-index-vector-stores-elasticsearch llama-index-embeddings-huggingface llama-index


# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("OllamaFunctionCallingAdapter API Key:")

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget -nv 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

vector_store = ElasticsearchStore(
    es_url="http://localhost:9200",  # see Elasticsearch Vector Store for more authentication options
    index_name="paul_graham_essay",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)