from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.packs.koda_retriever import KodaRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
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
# Koda Retriever: Quickstart

*For this example non-production ready infrastructure is leveraged, and default categories (and corresponding alpha values) are used.*

More specifically, the default sample data provided in a free start instance of [Pinecone](https://www.pinecone.io/) is used. This data consists of movie scripts and their summaries embedded in a free Pinecone vector database.

### Agenda:
- Setup
- Koda Retriever: Retrieval 
- Koda Retriever: Query Engine
"""
logger.info("# Koda Retriever: Quickstart")


"""
## Setup

Building *required objects* for a Koda Retriever.
- Vector Index
- LLM/Model

Other objects are *optional*, and will be used if provided:
- Reranker
- Custom categories & corresponding alpha weights
- A custom model trained on the custom info above
"""
logger.info("## Setup")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("sample-movies")

Settings.llm = MLXLlamaIndexLLMAdapter()
Settings.embed_model = MLXEmbedding()

vector_store = PineconeVectorStore(pinecone_index=index, text_key="summary")
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=Settings.embed_model
)

reranker = LLMRerank(llm=Settings.llm)  # optional

"""
## Building Koda Retriever
"""
logger.info("## Building Koda Retriever")

retriever = KodaRetriever(
    index=vector_index,
    llm=Settings.llm,
    reranker=reranker,  # optional
    verbose=True,
)

"""
## Retrieving w/ Koda Retriever
"""
logger.info("## Retrieving w/ Koda Retriever")

query = "How many Jurassic Park movies are there?"
results = retriever.retrieve(query)

results

"""
Those results don't look quite palletteable though. For that, lets look into making the response more *natural*. For that we'll likely need a Query Engine.

# Query Engine w/ Koda Retriever

Query Engines are [Llama Index abstractions](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html) that combine retrieval and synthesization of an LLM to interpret the results given by a retriever into a natural language response to the original query. They are themselves an end-to-end pipeline from query to natural langauge response.
"""
logger.info("# Query Engine w/ Koda Retriever")

query_engine = RetrieverQueryEngine.from_args(retriever=retriever)

response = query_engine.query(query)

str(response)

logger.info("\n\n[DONE]", bright=True)