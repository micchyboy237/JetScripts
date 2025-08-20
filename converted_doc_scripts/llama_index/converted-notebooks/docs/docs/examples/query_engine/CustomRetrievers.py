from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import QueryBundle
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
BaseRetriever,
VectorIndexRetriever,
KeywordTableSimpleRetriever,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/CustomRetrievers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Retriever Query Engine with Custom Retrievers - Simple Hybrid Search

In this tutorial, we show you how to define a very simple version of hybrid search! 

Combine keyword lookup retrieval with vector retrieval using "AND" and "OR" conditions.

### Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Retriever Query Engine with Custom Retrievers - Simple Hybrid Search")

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
### Download Data
"""
logger.info("### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Load Data

We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.
"""
logger.info("### Load Data")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


nodes = Settings.node_parser.get_nodes_from_documents(documents)


storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

"""
### Define Vector Index and Keyword Table Index over Same Data

We build a vector index and keyword index over the same DocumentStore
"""
logger.info("### Define Vector Index and Keyword Table Index over Same Data")


vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

"""
### Define Custom Retriever

We now define a custom retriever class that can implement basic hybrid search with both keyword lookup and semantic search.

- setting "AND" means we take the intersection of the two retrieved sets
- setting "OR" means we take the union
"""
logger.info("### Define Custom Retriever")





class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

"""
### Plugin Retriever into Query Engine

Plugin retriever into a query engine, and run some queries
"""
logger.info("### Plugin Retriever into Query Engine")


vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

response_synthesizer = get_response_synthesizer()

custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)
keyword_query_engine = RetrieverQueryEngine(
    retriever=keyword_retriever,
    response_synthesizer=response_synthesizer,
)

response = custom_query_engine.query(
    "What did the author do during his time at YC?"
)

logger.debug(response)

response = custom_query_engine.query(
    "What did the author do during his time at Yale?"
)

logger.debug(str(response))
len(response.source_nodes)

response = vector_query_engine.query(
    "What did the author do during his time at Yale?"
)

logger.debug(str(response))
len(response.source_nodes)

logger.info("\n\n[DONE]", bright=True)