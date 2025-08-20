from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionCache
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.ingestion.cache import RedisCache
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TransformComponent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import os
import re
import shutil
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/ingestion/advanced_ingestion_pipeline.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# %pip install llama-index-vector-stores-weaviate
# %pip install llama-index-embeddings-huggingface

# !pip install llama-index

"""
# Advanced Ingestion Pipeline

In this notebook, we implement an `IngestionPipeline` with the following features

- MongoDB transformation caching
- Automatic vector databse insertion
- A custom transformation

## Redis Cache Setup

All node + transformation combinations will have their outputs cached, which will save time on duplicate runs.
"""
logger.info("# Advanced Ingestion Pipeline")


ingest_cache = IngestionCache(
    cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
    collection="my_test_cache",
)

"""
## Vector DB Setup

For this example, we use weaviate as a vector store.
"""
logger.info("## Vector DB Setup")

# !pip install weaviate-client


auth_config = weaviate.AuthApiKey(api_key="...")

client = weaviate.Client(url="https://...", auth_client_secret=auth_config)


vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="CachingTest"
)

"""
## Transformation Setup
"""
logger.info("## Transformation Setup")


text_splitter = TokenTextSplitter(chunk_size=512)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

"""
### Custom Transformation
"""
logger.info("### Custom Transformation")



class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        return nodes

"""
## Running the pipeline
"""
logger.info("## Running the pipeline")


pipeline = IngestionPipeline(
    transformations=[
        TextCleaner(),
        text_splitter,
        embed_model,
        TitleExtractor(),
    ],
    vector_store=vector_store,
    cache=ingest_cache,
)


documents = SimpleDirectoryReader("./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

nodes = pipeline.run(documents=documents)

"""
## Using our populated vector store
"""
logger.info("## Using our populated vector store")


# os.environ["OPENAI_API_KEY"] = "sk-..."


index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

query_engine = index.as_query_engine()

logger.debug(query_engine.query("What did the author do growing up?"))

"""
## Re-run Ingestion to test Caching

The next code block will execute almost instantly due to caching.
"""
logger.info("## Re-run Ingestion to test Caching")

pipeline = IngestionPipeline(
    transformations=[TextCleaner(), text_splitter, embed_model],
    cache=ingest_cache,
)

nodes = pipeline.run(documents=documents)

"""
## Clear the cache
"""
logger.info("## Clear the cache")

ingest_cache.clear()

logger.info("\n\n[DONE]", bright=True)