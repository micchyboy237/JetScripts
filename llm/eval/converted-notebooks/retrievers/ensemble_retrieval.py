from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/ensemble_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Ensemble Retrieval Guide
# 
# Oftentimes when building a RAG applications there are many retreival parameters/strategies to decide from (from chunk size to vector vs. keyword vs. hybrid search, for instance).
# 
# Thought: what if we could try a bunch of strategies at once, and have any AI/reranker/LLM prune the results?
# 
# This achieves two purposes:
# - Better (albeit more costly) retrieved results by pooling results from multiple strategies, assuming the reranker is good
# - A way to benchmark different retrieval strategies against each other (w.r.t reranker)
# 
# This guide showcases this over the Llama 2 paper. We do ensemble retrieval over different chunk sizes and also different indices.
# 
# **NOTE**: A closely related guide is our [Ensemble Query Engine Guide](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/ensemble_qury_engine.html) - make sure to check it out!

# %pip install llama-index-llms-ollama
# %pip install llama-index-postprocessor-cohere-rerank
# %pip install llama-index-readers-file pymupdf

# %load_ext autoreload
# %autoreload 2

## Setup
# 
# Here we define the necessary imports.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# !pip install llama-index

import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core import SummaryIndex
from llama_index.core.response.notebook_utils import display_response
from llama_index.llms.ollama import Ollama

## Load Data
# 
# In this section we first load in the Llama 2 paper as a single document. We then chunk it multiple times, according to different chunk sizes. We build a separate vector index corresponding to each chunk size.

# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

from pathlib import Path
from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader

loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))
doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

# Here we try out different chunk sizes: 128, 256, 512, and 1024.

llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)
chunk_sizes = [128, 256, 512, 1024]
nodes_list = []
vector_indices = []
for chunk_size in chunk_sizes:
    print(f"Chunk Size: {chunk_size}")
    splitter = SentenceSplitter(chunk_size=chunk_size)
    nodes = splitter.get_nodes_from_documents(docs)

    for node in nodes:
        node.metadata["chunk_size"] = chunk_size
        node.excluded_embed_metadata_keys = ["chunk_size"]
        node.excluded_llm_metadata_keys = ["chunk_size"]

    nodes_list.append(nodes)

    vector_index = VectorStoreIndex(nodes)
    vector_indices.append(vector_index)

## Define Ensemble Retriever
# 
# We setup an "ensemble" retriever primarily using our recursive retrieval abstraction. This works like the following:
# - Define a separate `IndexNode` corresponding to the vector retriever for each chunk size (retriever for chunk size 128, retriever for chunk size 256, and more)
# - Put all IndexNodes into a single `SummaryIndex` - when the corresponding retriever is called, *all* nodes are returned.
# - Define a Recursive Retriever, with the root node being the summary index retriever. This will first fetch all nodes from the summary index retriever, and then recursively call the vector retriever for each chunk size.
# - Rerank the final results.
# 
# The end result is that all vector retrievers are called when a query is run.

from llama_index.core.tools import RetrieverTool
from llama_index.core.schema import IndexNode

retriever_dict = {}
retriever_nodes = []
for chunk_size, vector_index in zip(chunk_sizes, vector_indices):
    node_id = f"chunk_{chunk_size}"
    node = IndexNode(
        text=(
            "Retrieves relevant context from the Llama 2 paper (chunk size"
            f" {chunk_size})"
        ),
        index_id=node_id,
    )
    retriever_nodes.append(node)
    retriever_dict[node_id] = vector_index.as_retriever()

# Define recursive retriever.

from llama_index.core.selectors import PydanticMultiSelector

from llama_index.core.retrievers import RouterRetriever
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core import SummaryIndex

summary_index = SummaryIndex(retriever_nodes)

retriever = RecursiveRetriever(
    root_id="root",
    retriever_dict={"root": summary_index.as_retriever(), **retriever_dict},
)

# Let's test the retriever on a sample query.

nodes = await retriever.aretrieve(
    "Tell me about the main aspects of safety fine-tuning"
)

print(f"Number of nodes: {len(nodes)}")
for node in nodes:
    print(node.node.metadata["chunk_size"])
    print(node.node.get_text())

# Define reranker to process the final retrieved set of nodes.

from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(top_n=10)

# Define retriever query engine to integrate the recursive retriever + reranker together.

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine(retriever, node_postprocessors=[reranker])

response = query_engine.query(
    "Tell me about the main aspects of safety fine-tuning"
)

display_response(
    response, show_source=True, source_length=500, show_source_metadata=True
)

### Analyzing the Relative Importance of each Chunk
# 
# One interesting property of ensemble-based retrieval is that through reranking, we can actually use the ordering of chunks in the final retrieved set to determine the importance of each chunk size. For instance, if certain chunk sizes are always ranked near the top, then those are probably more relevant to the query.

from collections import defaultdict
import pandas as pd


def mrr_all(metadata_values, metadata_key, source_nodes):
    value_to_mrr_dict = {}
    for metadata_value in metadata_values:
        mrr = 0
        for idx, source_node in enumerate(source_nodes):
            if source_node.node.metadata[metadata_key] == metadata_value:
                mrr = 1 / (idx + 1)
                break
            else:
                continue

        value_to_mrr_dict[metadata_value] = mrr

    df = pd.DataFrame(value_to_mrr_dict, index=["MRR"])
    df.style.set_caption("Mean Reciprocal Rank")
    return df

print("Mean Reciprocal Rank for each Chunk Size")
mrr_all(chunk_sizes, "chunk_size", response.source_nodes)

## Evaluation
# 
# We more rigorously evaluate how well an ensemble retriever works compared to the "baseline" retriever.
# 
# We define/load an eval benchmark dataset and then run different evaluations over it.
# 
# **WARNING**: This can be *expensive*, especially with GPT-4. Use caution and tune the sample size to fit your budget.

from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.llms.ollama import Ollama
import nest_asyncio

nest_asyncio.apply()

eval_llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)
dataset_generator = DatasetGenerator(
    nodes_list[-1],
    llm=eval_llm,
    show_progress=True,
    num_questions_per_chunk=2,
)

eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=60)

eval_dataset.save_json("data/llama2_eval_qr_dataset.json")

eval_dataset = QueryResponseDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

### Compare Results

import asyncio
import nest_asyncio

nest_asyncio.apply()

from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    PairwiseComparisonEvaluator,
)

evaluator_c = CorrectnessEvaluator(llm=eval_llm)
evaluator_s = SemanticSimilarityEvaluator(llm=eval_llm)
evaluator_r = RelevancyEvaluator(llm=eval_llm)
evaluator_f = FaithfulnessEvaluator(llm=eval_llm)

pairwise_evaluator = PairwiseComparisonEvaluator(llm=eval_llm)

from llama_index.core.evaluation.eval_utils import (
    get_responses,
    get_results_df,
)
from llama_index.core.evaluation import BatchEvalRunner

max_samples = 60

eval_qs = eval_dataset.questions
qr_pairs = eval_dataset.qr_pairs
ref_response_strs = [r for (_, r) in qr_pairs]

base_query_engine = vector_indices[-1].as_query_engine(similarity_top_k=2)
reranker = CohereRerank(top_n=4)
query_engine = RetrieverQueryEngine(retriever, node_postprocessors=[reranker])

base_pred_responses = get_responses(
    eval_qs[:max_samples], base_query_engine, show_progress=True
)

pred_responses = get_responses(
    eval_qs[:max_samples], query_engine, show_progress=True
)

import numpy as np

pred_response_strs = [str(p) for p in pred_responses]
base_pred_response_strs = [str(p) for p in base_pred_responses]

evaluator_dict = {
    "correctness": evaluator_c,
    "faithfulness": evaluator_f,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=1, show_progress=True)

eval_results = batch_runner.evaluate_responses(
    queries=eval_qs[:max_samples],
    responses=pred_responses[:max_samples],
    reference=ref_response_strs[:max_samples],
)

base_eval_results = batch_runner.evaluate_responses(
    queries=eval_qs[:max_samples],
    responses=base_pred_responses[:max_samples],
    reference=ref_response_strs[:max_samples],
)

results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Ensemble Retriever", "Base Retriever"],
    ["correctness", "faithfulness", "semantic_similarity"],
)
display(results_df)

batch_runner = BatchEvalRunner(
    {"pairwise": pairwise_evaluator}, workers=3, show_progress=True
)

pairwise_eval_results = batch_runner.evaluate_response_strs(
    queries=eval_qs[:max_samples],
    response_strs=pred_response_strs[:max_samples],
    reference=base_pred_response_strs[:max_samples],
)

results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Ensemble Retriever", "Base Retriever"],
    ["pairwise"],
)
display(results_df)

logger.info("\n\n[DONE]", bright=True)