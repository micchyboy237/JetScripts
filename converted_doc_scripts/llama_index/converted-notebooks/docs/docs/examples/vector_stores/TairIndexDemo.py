from IPython.display import Markdown, display
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
GPTVectorStoreIndex,
SimpleDirectoryReader,
Document,
)
from llama_index.core import StorageContext
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.tair import TairVectorStore
import logging
import os
import shutil
import sys
import textwrap
import warnings


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/TairIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Tair Vector Store

In this notebook we are going to show a quick demo of using the TairVectorStore.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Tair Vector Store")

# %pip install llama-index-vector-stores-tair

# !pip install llama-index



warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"



"""
### Setup MLX
Lets first begin by adding the openai api key. This will allow us to access openai for embeddings and to use chatgpt.
"""
logger.info("### Setup MLX")


# os.environ["OPENAI_API_KEY"] = "sk-<your key here>"

"""
### Download Data
"""
logger.info("### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Read in a dataset
"""
logger.info("### Read in a dataset")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
logger.debug(
    "Document ID:",
    documents[0].doc_id,
    "Document Hash:",
    documents[0].doc_hash,
)

"""
### Build index from documents
Let's build a vector index with ``GPTVectorStoreIndex``, using ``TairVectorStore`` as its backend. Replace ``tair_url`` with the actual url of your Tair instance.
"""
logger.info("### Build index from documents")


tair_url = "redis://{username}:{password}@r-bp****************.redis.rds.aliyuncs.com:{port}"

vector_store = TairVectorStore(
    tair_url=tair_url, index_name="pg_essays", overwrite=True
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = GPTVectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query the data

Now we can use the index as knowledge base and ask questions to it.
"""
logger.info("### Query the data")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author learn?")
logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("What was a hard moment for the author?")
logger.debug(textwrap.fill(str(response), 100))

"""
### Deleting documents
To delete a document from the index, use `delete` method.
"""
logger.info("### Deleting documents")

document_id = documents[0].doc_id
document_id

info = vector_store.client.tvs_get_index("pg_essays")
logger.debug("Number of documents", int(info["data_count"]))

vector_store.delete(document_id)

info = vector_store.client.tvs_get_index("pg_essays")
logger.debug("Number of documents", int(info["data_count"]))

"""
### Deleting index
Delete the entire index using `delete_index` method.
"""
logger.info("### Deleting index")

vector_store.delete_index()

logger.debug("Check index existence:", vector_store.client._index_exists())

logger.info("\n\n[DONE]", bright=True)