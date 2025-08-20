from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
SimpleDirectoryReader,
load_index_from_storage,
VectorStoreIndex,
StorageContext,
)
from llama_index.core import Settings
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import (
MetadataFilters,
MetadataFilter,
)
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.vector_stores.oceanbase import OceanBaseVectorStore
from pyobvector import ObVecClient
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/OceanBaseVectorStore.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# OceanBase Vector Store

> [OceanBase Database](https://github.com/oceanbase/oceanbase) is a distributed relational database. It is developed entirely by Ant Group. The OceanBase Database is built on a common server cluster. Based on the Paxos protocol and its distributed structure, the OceanBase Database provides high availability and linear scalability. The OceanBase Database is not dependent on specific hardware architectures.

This notebook describes in detail how to use the OceanBase vector store functionality in LlamaIndex.

#### Setup
"""
logger.info("# OceanBase Vector Store")

# %pip install llama-index-vector-stores-oceanbase
# %pip install llama-index
# %pip install llama-index-embeddings-dashscope
# %pip install llama-index-llms-dashscope

"""
#### Deploy a standalone OceanBase server with docker
"""
logger.info("#### Deploy a standalone OceanBase server with docker")

# %docker run --name=ob433 -e MODE=slim -p 2881:2881 -d oceanbase/oceanbase-ce:4.3.3.0-100000142024101215

"""
#### Creating ObVecClient
"""
logger.info("#### Creating ObVecClient")


client = ObVecClient()
client.perform_raw_text_sql(
    "ALTER SYSTEM ob_vector_memory_limit_percentage = 30"
)

"""
Config dashscope embedding model and LLM.
"""
logger.info("Config dashscope embedding model and LLM.")


Settings.embed_model = DashScopeEmbedding()


dashscope_llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
)

"""
#### Load documents
"""
logger.info("#### Load documents")


"""
Download Data & Load Data
"""
logger.info("Download Data & Load Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

oceanbase = OceanBaseVectorStore(
    client=client,
    dim=1536,
    drop_old=True,
    normalize=True,
)

storage_context = StorageContext.from_defaults(vector_store=oceanbase)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
#### Query Index
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine(llm=dashscope_llm)
res = query_engine.query("What did the author do growing up?")
res.response

"""
#### Metadata Filtering

OceanBase Vector Store supports metadata filtering in the form of `=`、 `>`、`<`、`!=`、`>=`、`<=`、`in`、`not in`、`like`、`IS NULL` at query time.
"""
logger.info("#### Metadata Filtering")


query_engine = index.as_query_engine(
    llm=dashscope_llm,
    filters=MetadataFilters(
        filters=[
            MetadataFilter(key="book", value="paul_graham", operator="!="),
        ]
    ),
    similarity_top_k=10,
)

res = query_engine.query("What did the author learn?")
res.response

"""
#### Delete Documents
"""
logger.info("#### Delete Documents")

oceanbase.delete(documents[0].doc_id)

query_engine = index.as_query_engine(llm=dashscope_llm)
res = query_engine.query("What did the author do growing up?")
res.response

logger.info("\n\n[DONE]", bright=True)