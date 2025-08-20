from IPython.display import Markdown, display
from google.colab import userdata
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
StorageContext,
Settings,
)
from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
BasePydanticVectorStore,
MetadataFilters,
VectorStoreQuery,
VectorStoreQueryMode,
VectorStoreQueryResult,
)
from llama_index.core.vector_stores.types import (
MetadataFilter,
MetadataFilters,
FilterOperator,
FilterCondition,
)
from llama_index.core.vector_stores.utils import (
DEFAULT_TEXT_KEY,
legacy_metadata_dict_to_node,
metadata_dict_to_node,
node_to_metadata_dict,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from moorcheh_sdk import MoorchehClient
from typing import Any, Callable, Dict, List, Optional, cast
import logging
import os
import shutil
import sys


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
# Moorcheh Vector Store Demo

## Install Required Packages
"""
logger.info("# Moorcheh Vector Store Demo")

# !pip install llama_index
# !pip install moorcheh_sdk

"""
## Import Required Libraries
"""
logger.info("## Import Required Libraries")


"""
##  Configure Logging
"""
logger.info("##  Configure Logging")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
## Load Moorcheh API Key
"""
logger.info("## Load Moorcheh API Key")


api_key = os.environ["MOORCHEH_API_KEY"] = userdata.get("MOORCHEH_API_KEY")

if "MOORCHEH_API_KEY" not in os.environ:
    raise EnvironmentError(f"Environment variable MOORCHEH_API_KEY is not set")

"""
## Load and Chunk Documents
"""
logger.info("## Load and Chunk Documents")

documents = SimpleDirectoryReader("./documents").load_data()

Settings.chunk_size = 1024
Settings.chunk_overlap = 20

"""
## Initialize Vector Store and Create Index
"""
logger.info("## Initialize Vector Store and Create Index")

__all__ = ["MoorchehVectorStore"]

vector_store = MoorchehVectorStore(
    api_key=api_key,
    namespace="llamaindex_moorcheh",
    namespace_type="text",
    vector_dimension=None,
    add_sparse_vector=False,
    batch_size=100,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
## Query the Vector Store
"""
logger.info("## Query the Vector Store")

query_engine = index.as_query_engine()
response = vector_store.generate_answer(
    query="Which company has had the highest revenue in 2025 and why?"
)
moorcheh_response = vector_store.get_generative_answer(
    query="Which company has had the highest revenue in 2025 and why?",
    ai_model="anthropic.claude-3-7-sonnet-20250219-v1:0",
)

display(Markdown(f"<b>{response}</b>"))
logger.debug(
    "\n\n================================\n\n",
    response,
    "\n\n================================\n\n",
)
logger.debug(
    "\n\n================================\n\n",
    moorcheh_response,
    "\n\n================================\n\n",
)

filter = MetadataFilters(
    filters=[
        MetadataFilter(
            key="file_path",
            value="insert the file path to the document here",
            operator=FilterOperator.EQ,
        )
    ],
    condition=FilterCondition.AND,
)

logger.info("\n\n[DONE]", bright=True)