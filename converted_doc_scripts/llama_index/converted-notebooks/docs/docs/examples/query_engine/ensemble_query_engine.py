import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.selectors import (
PydanticMultiSelector,
PydanticSingleSelector,
)
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/ensemble_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Ensemble Query Engine Guide

Oftentimes when building a RAG application there are different query pipelines you need to experiment with (e.g. top-k retrieval, keyword search, knowledge graphs).

Thought: what if we could try a bunch of strategies at once, and have the LLM 1) rate the relevance of each query, and 2) synthesize the results?

This guide showcases this over the Great Gatsby. We do ensemble retrieval over different chunk sizes and also different indices.

**NOTE**: Please also see our closely-related [Ensemble Retrieval Guide](https://gpt-index.readthedocs.io/en/stable/examples/retrievers/ensemble_retrieval.html)!

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Ensemble Query Engine Guide")

# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
## Setup
"""
logger.info("## Setup")

# import nest_asyncio

# nest_asyncio.apply()

"""
## Download Data
"""
logger.info("## Download Data")

# !wget 'https://raw.githubusercontent.com/jerryjliu/llama_index/main/examples/gatsby/gatsby_full.txt' -O 'gatsby_full.txt'

"""
## Load Data

We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.
"""
logger.info("## Load Data")



documents = SimpleDirectoryReader(
    input_files=["./gatsby_full.txt"]
).load_data()

"""
## Define Query Engines
"""
logger.info("## Define Query Engines")


Settings.llm = MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
Settings.chunk_size = 1024

nodes = Settings.node_parser.get_nodes_from_documents(documents)


storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


keyword_index = SimpleKeywordTableIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True,
)
vector_index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True,
)


QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question. If the answer is not in the context, inform "
    "the user that you can't answer the question - DO NOT MAKE UP AN ANSWER.\n"
    "In addition to returning the answer, also return a relevance score as to "
    "how relevant the answer is to the question. "
    "Question: {query_str}\n"
    "Answer (including relevance score): "
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

keyword_query_engine = keyword_index.as_query_engine(
    text_qa_template=QA_PROMPT
)
vector_query_engine = vector_index.as_query_engine(text_qa_template=QA_PROMPT)

response = vector_query_engine.query(
    "Describe and summarize the interactions between Gatsby and Daisy"
)

logger.debug(response)

response = keyword_query_engine.query(
    "Describe and summarize the interactions between Gatsby and Daisy"
)

logger.debug(response)

"""
## Define Router Query Engine
"""
logger.info("## Define Router Query Engine")



keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_query_engine,
    description="Useful for answering questions about this essay",
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for answering questions about this essay",
)


TREE_SUMMARIZE_PROMPT_TMPL = (
    "Context information from multiple sources is below. Each source may or"
    " may not have \na relevance score attached to"
    " it.\n---------------------\n{context_str}\n---------------------\nGiven"
    " the information from multiple sources and their associated relevance"
    " scores (if provided) and not prior knowledge, answer the question. If"
    " the answer is not in the context, inform the user that you can't answer"
    " the question.\nQuestion: {query_str}\nAnswer: "
)

tree_summarize = TreeSummarize(
    summary_template=PromptTemplate(TREE_SUMMARIZE_PROMPT_TMPL)
)

query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[
        keyword_tool,
        vector_tool,
    ],
    summarizer=tree_summarize,
)

"""
## Experiment with Queries
"""
logger.info("## Experiment with Queries")

async def async_func_0():
    response = query_engine.query(
        "Describe and summarize the interactions between Gatsby and Daisy"
    )
    return response
response = asyncio.run(async_func_0())
logger.success(format_json(response))
logger.debug(response)

response.source_nodes

async def async_func_7():
    response = query_engine.query(
        "What part of his past is Gatsby trying to recapture?"
    )
    return response
response = asyncio.run(async_func_7())
logger.success(format_json(response))
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)