from IPython.display import Markdown, display
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import openai
import os
import qdrant_client
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/pinecone_metadata_filter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Qdrant Vector Store - Default Qdrant Filters

Example on how to use Filters from the qdrant_client SDK directly in your Retriever / Query Engine
"""
logger.info("# Qdrant Vector Store - Default Qdrant Filters")

# %pip install llama-index-vector-stores-qdrant

# !pip3 install llama-index qdrant_client


client = qdrant_client.QdrantClient(location=":memory:")

nodes = [
    TextNode(
        text="りんごとは",
        metadata={"author": "Tanaka", "fruit": "apple", "city": "Tokyo"},
    ),
    TextNode(
        text="Was ist Apfel?",
        metadata={"author": "David", "fruit": "apple", "city": "Berlin"},
    ),
    TextNode(
        text="Orange like the sun",
        metadata={"author": "Jane", "fruit": "orange", "city": "Hong Kong"},
    ),
    TextNode(
        text="Grape is...",
        metadata={"author": "Jane", "fruit": "grape", "city": "Hong Kong"},
    ),
    TextNode(
        text="T-dot > G-dot",
        metadata={"author": "George", "fruit": "grape", "city": "Toronto"},
    ),
    TextNode(
        text="6ix Watermelons",
        metadata={
            "author": "George",
            "fruit": "watermelon",
            "city": "Toronto",
        },
    ),
]

openai.api_key = "YOUR_API_KEY"
vector_store = QdrantVectorStore(
    client=client, collection_name="fruit_collection"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)



filters = Filter(
    should=[
        Filter(
            must=[
                FieldCondition(
                    key="fruit",
                    match=MatchValue(value="apple"),
                ),
                FieldCondition(
                    key="city",
                    match=MatchValue(value="Tokyo"),
                ),
            ]
        ),
        Filter(
            must=[
                FieldCondition(
                    key="fruit",
                    match=MatchValue(value="grape"),
                ),
                FieldCondition(
                    key="city",
                    match=MatchValue(value="Toronto"),
                ),
            ]
        ),
    ]
)

retriever = index.as_retriever(vector_store_kwargs={"qdrant_filters": filters})

response = retriever.retrieve("Who makes grapes?")
for node in response:
    logger.debug("node", node.score)
    logger.debug("node", node.text)
    logger.debug("node", node.metadata)

logger.info("\n\n[DONE]", bright=True)