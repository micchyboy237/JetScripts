import os
from jet.cache.joblib.utils import load_persistent_cache, save_persistent_cache, ttl_cache
from jet.file.utils import save_file
from pydantic import BaseModel, Field
from jet.logger import logger
from jet.token.token_utils import get_model_max_tokens
from jet.utils.commands import copy_to_clipboard
from llama_index.core import SimpleDirectoryReader
from llama_index.core.postprocessor import StructuredLLMRerank
from jet.llm.ollama.base import Ollama, OllamaEmbedding, VectorStoreIndex
from IPython.display import Markdown, display
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
import pandas as pd
from IPython.display import display, HTML
from copy import deepcopy

from llama_index.core.schema import NodeWithScore

model = "llama3.2"
embed_model = "mxbai-embed-large"

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/Structured-LLMReranker-Lyft-10k.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Structured LLM Reranker Demonstration (2021 Lyft 10-k)

This tutorial showcases how to do a two-stage pass for retrieval. Use embedding-based retrieval with a high top-k value
in order to maximize recall and get a large set of candidate items. Then, use LLM-based retrieval
to dynamically select the nodes that are actually relevant to the query using structured output.

Usage of `StructuredLLMReranker` is preferred over `LLMReranker` when you are using a model that supports function calling.
This class will make use of the structured output capability of the model instead of relying on prompting the model to rank the nodes in a desired format.
"""

# %pip install llama-index-llms-ollama

# import nest_asyncio

# nest_asyncio.apply()


"""
## Download Data
"""

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

"""
## Load Data, Build Index
"""

DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/llama_index/docs/docs/examples/data/10k/lyft_2021.pdf"
CACHE_FILE = "nodes_lyft_2021.pkl"

Settings.llm = Ollama(temperature=0, model=model,
                      request_timeout=300.0, context_window=get_model_max_tokens(model))
Settings.embed_model = OllamaEmbedding(model_name=embed_model)

Settings.chunk_overlap = 0
Settings.chunk_size = 128


load_persistent_cache(CACHE_FILE)

if CACHE_FILE in ttl_cache:
    documents = ttl_cache[CACHE_FILE]
    logger.success(f"Cache hit: {CACHE_FILE} - Length: {len(documents)}")
else:
    documents = SimpleDirectoryReader(
        input_files=[DATA_FILE]
    ).load_data()
    ttl_cache[CACHE_FILE] = documents
    save_persistent_cache(CACHE_FILE)

documents = [doc for doc in documents if "covid" in doc.text.lower()]

index = VectorStoreIndex.from_documents(
    documents,
)


"""
## Custom output cls structure
"""


class DocumentWithRelevance(BaseModel):
    """Document rankings as selected by model."""

    document_number: int = Field(
        description="The number of the document within the provided list"
    )
    relevance: int = Field(
        description="Relevance score from 1-10 of the document to the given query - based on the document content",
        json_schema_extra={"minimum": 1, "maximum": 10},
    )
    feedback: str = Field(
        description="Brief feedback on the document's relevance. Example: 'Highly relevant - directly discusses Lyft's COVID-19 safety protocols and driver support measures'",
        min_length=1, max_length=100
    )


class DocumentRelevanceList(BaseModel):
    """List of documents with relevance scores."""

    documents: list[DocumentWithRelevance] = Field(
        description="List of documents with relevance scores"
    )
    feedback: str = Field(
        description="Overall feedback on the relevance of all documents. Example: 'Found 2 relevant documents discussing Lyft's COVID-19 response. Document 1 is highly detailed while Document 2 only has a brief mention.'",
        min_length=1,
        max_length=200
    )


document_relevance_list_cls = DocumentRelevanceList


"""
## Retrieval Comparisons
"""


def get_retrieved_nodes(
    query_str, vector_top_k=10, reranker_top_n=5, with_reranker=False, choice_batch_size=10
):
    query_bundle = QueryBundle(query_str)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        reranker = StructuredLLMRerank(
            llm=Settings.llm,
            choice_batch_size=choice_batch_size,
            top_n=reranker_top_n,
            document_relevance_list_cls=document_relevance_list_cls
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes


def visualize_retrieved_nodes(nodes: list[NodeWithScore]):
    results = [{"score": node.score, "text": node.text} for node in nodes]
    copy_to_clipboard(results)
    logger.pretty(results)
    return results


# reranked_nodes = get_retrieved_nodes(
#     "What is Lyft's response to COVID-19?", vector_top_k=5, with_reranker=False
# )

# visualize_retrieved_nodes(reranked_nodes)

# reranked_nodes = get_retrieved_nodes(
#     "What is Lyft's response to COVID-19?",
#     vector_top_k=20,
#     reranker_top_n=5,
#     with_reranker=True,
# )

# visualize_retrieved_nodes(reranked_nodes)

# reranked_nodes = get_retrieved_nodes(
#     "What initiatives are the company focusing on independently of COVID-19?",
#     vector_top_k=5,
#     with_reranker=False,
# )

# visualize_retrieved_nodes(reranked_nodes)

query = "What initiatives are the company focusing on independently of COVID-19?"
reranked_nodes = get_retrieved_nodes(
    query,
    vector_top_k=40,
    reranker_top_n=10,
    with_reranker=True,
)


results = visualize_retrieved_nodes(reranked_nodes)

output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/llama_index/node_postprocessor/generated/structured_llm_reranker_results.json"
save_file({
    "query": query,
    "results": results
}, output_file)

logger.info("\n\n[DONE]", bright=True)
