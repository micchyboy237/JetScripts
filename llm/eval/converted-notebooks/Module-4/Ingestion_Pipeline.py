import re
from llama_index.core.schema import TransformComponent
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.extractors import TitleExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
import os
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Ingestion Pipeline
#
# In this notebook we will demonstrate usage of Ingestion Pipeline in building RAG applications.
#
# [Ingestion Pipeline](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/)

# Installation

# !pip install llama-index llama-index-vector-stores-qdrant

# Set API Key


nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "sk-..."

# Download Data

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# Load Data


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries",
    required_exts=["*.md"]
).load_data()

# Ingestion Pipeline - Apply Transformations


# Text Splitters

pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
    ]
)
nodes = pipeline.run(documents=documents)

nodes[0]

# Text Splitter + Metadata Extractor

pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
    ]
)
nodes = pipeline.run(documents=documents)

nodes[0].metadata["document_title"]

# Text Splitter + Metadata Extractor + Ollama Embedding

pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
        OllamaEmbedding(model_name="mxbai-embed-large"),
    ]
)
nodes = pipeline.run(documents=documents)

nodes[0].metadata["document_title"]

nodes[0]

# Cache

pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
    ]
)
nodes = pipeline.run(documents=documents)

pipeline.cache.persist("./llama_cache.json")
new_cache = IngestionCache.from_persist_path("./llama_cache.json")

new_pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
    ],
    cache=new_cache,
)

# Now it will run instantly due to the cache.
#
# Will be very useful when extracting metadata and also creating embeddings

nodes = new_pipeline.run(documents=documents)

# Now let's add embeddings to it. You will observe that the parsing of nodes, title extraction is loaded from cache and Ollama embeddings are created now.

pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
        OllamaEmbedding(model_name="mxbai-embed-large"),
    ],
    cache=new_cache,
)
nodes = pipeline.run(documents=documents)

pipeline.cache.persist("./nodes_embedding.json")
nodes_embedding_cache = IngestionCache.from_persist_path(
    "./nodes_embedding.json"
)

pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
        OllamaEmbedding(model_name="mxbai-embed-large"),
    ],
    cache=nodes_embedding_cache,
)

nodes = pipeline.run(documents=documents)

nodes[0].text

# RAG using Ingestion Pipeline


client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(
    client=client, collection_name="llama_index_vector_store"
)
pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
        OllamaEmbedding(model_name="mxbai-embed-large"),
    ],
    cache=nodes_embedding_cache,
    vector_store=vector_store,
)
nodes = pipeline.run(documents=documents)


index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine()

response = query_engine.query("What did paul graham do growing up?")

print(response)

# Custom Transformations
#
# Implementing custom transformations is pretty easy.
#
# Let's include a transformation that removes special characters from the text before generating embeddings.
#
# The primary requirement for transformations is that they should take a list of nodes as input and return a modified list of nodes.


class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        return nodes


pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
        TextCleaner(),
        OllamaEmbedding(model_name="mxbai-embed-large"),
    ],
    cache=nodes_embedding_cache,
)

nodes = pipeline.run(documents=documents)

nodes[0].text

logger.info("\n\n[DONE]", bright=True)
