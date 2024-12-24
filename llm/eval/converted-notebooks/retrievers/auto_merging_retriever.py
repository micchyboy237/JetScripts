import numpy as np
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation.eval_utils import (
    get_responses,
    get_results_df,
)
import pandas as pd
from collections import defaultdict
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    PairwiseComparisonEvaluator,
)
import asyncio
import nest_asyncio
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
)
from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader
from pathlib import Path
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
settings = initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/auto_merging_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Auto Merging Retriever
#
# In this notebook, we showcase our `AutoMergingRetriever`, which looks at a set of leaf nodes and recursively "merges" subsets of leaf nodes that reference a parent node beyond a given threshold. This allows us to consolidate potentially disparate, smaller contexts into a larger context that might help synthesis.
#
# You can define this hierarchy yourself over a set of documents, or you can make use of our brand-new text parser: a HierarchicalNodeParser that takes in a candidate set of documents and outputs an entire hierarchy of nodes, from "coarse-to-fine".

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-file pymupdf

# %load_ext autoreload
# %autoreload 2

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# !pip install llama-index

# Load Data
#
# Let's first load the Llama 2 paper: https://arxiv.org/pdf/2307.09288.pdf. This will be our test data.

# !mkdir -p 'data/'
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"


# loader = PyMuPDFReader()
# docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

# By default, the PDF reader creates a separate doc for each page.
# For the sake of this notebook, we stitch docs together into one doc.
# This will help us better highlight auto-merging capabilities that "stitch" chunks together later on.


docs = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume").load_data()
# num_questions_per_chunk = 1
question_gen_query = f"""
You are a job interviewer. Your task is to setup questions or an upcoming interview. You are provided with applicant resume information. Restrict the questions to the context information provided.
""".strip()
query = "What is your work experience?"

# Parse Chunk Hierarchy from Text, Load into Storage
#
# In this section we make use of the `HierarchicalNodeParser`. This will output a hierarchy of nodes, from top-level nodes with bigger chunk sizes to child nodes with smaller chunk sizes, where each child node has a parent node with a bigger chunk size.
#
# By default, the hierarchy is:
# - 1st level: chunk size 2048
# - 2nd level: chunk size 512
# - 3rd level: chunk size 128
#
#
# We then load these nodes into storage. The leaf nodes are indexed and retrieved via a vector store - these are the nodes that will first be directly retrieved via similarity search. The other nodes will be retrieved from a docstore.


node_parser = HierarchicalNodeParser.from_defaults()

nodes = node_parser.get_nodes_from_documents(docs)

logger.log("nodes:", len(nodes), colors=["GRAY", "INFO"])

# Here we import a simple helper function for fetching "leaf" nodes within a node list.
# These are nodes that don't have children of their own.


leaf_nodes = get_leaf_nodes(nodes)

logger.log("leaf_nodes:", len(leaf_nodes), colors=["GRAY", "INFO"])

root_nodes = get_root_nodes(nodes)

# Load into Storage
#
# We define a docstore, which we load all nodes into.
#
# We then define a `VectorStoreIndex` containing just the leaf-level nodes.


docstore = SimpleDocumentStore()

docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)


base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)

# Define Retriever


base_retriever = base_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)


nodes = retriever.retrieve(query)
base_nodes = base_retriever.retrieve(query)

logger.log("Query:", query, colors=["WHITE", "INFO"])
logger.log("Node Scores (AutoMergingRetriever):", f"({len(nodes)})", colors=[
           "WHITE", "SUCCESS"])
for node in nodes:
    logger.newline()
    logger.log("Score:", f"{node.score:.2f}", colors=["WHITE", "SUCCESS"])
    logger.log("File:", node.metadata['file_path'], colors=["WHITE", "DEBUG"])

len(base_nodes)

logger.log("Query:", query, colors=["WHITE", "INFO"])
logger.log("Node Scores (BaseRetriever):", f"({len(base_nodes)})", colors=[
           "WHITE", "SUCCESS"])
for node in base_nodes:
    logger.newline()
    logger.log("Score:", f"{node.score:.2f}", colors=["WHITE", "SUCCESS"])
    logger.log("File:", node.metadata['file_path'], colors=["WHITE", "DEBUG"])


# Plug it into Query Engine


query_engine = RetrieverQueryEngine.from_args(retriever)
base_query_engine = RetrieverQueryEngine.from_args(base_retriever)

response = query_engine.query(query)

print(str(response))

base_response = base_query_engine.query(query)

print(str(base_response))

# Evaluation
#
# We evaluate how well the hierarchical retriever works compared to the baseline retriever in a more quantitative manner.
#
# **WARNING**: This can be *expensive*, especially with GPT-4. Use caution and tune the sample size to fit your budget.


nest_asyncio.apply()

eval_llm = Ollama(model="llama3.1", request_timeout=1200.0,
                  context_window=4096)
dataset_generator = DatasetGenerator(
    root_nodes[:20],
    llm=eval_llm,
    show_progress=True,
    # num_questions_per_chunk=num_questions_per_chunk,
    question_gen_query=question_gen_query,
)

eval_dataset = dataset_generator.generate_dataset_from_nodes(num=60)

eval_dataset.save_json("data/llama2_eval_qr_dataset.json")

eval_dataset = QueryResponseDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

# Compare Results
#
# We run evaluations on each of the retrievers: correctness, semantic similarity, relevance, and faithfulness.


nest_asyncio.apply()


evaluator_c = CorrectnessEvaluator(llm=eval_llm)
evaluator_s = SemanticSimilarityEvaluator(embed_model=settings.embed_model)
evaluator_r = RelevancyEvaluator(llm=eval_llm)
evaluator_f = FaithfulnessEvaluator(llm=eval_llm)


eval_qs = eval_dataset.questions
qr_pairs = eval_dataset.qr_pairs
ref_response_strs = [r for (_, r) in qr_pairs]

pred_responses = get_responses(eval_qs, query_engine, show_progress=True)

base_pred_responses = get_responses(
    eval_qs, base_query_engine, show_progress=True
)


pred_response_strs = [str(p) for p in pred_responses]
base_pred_response_strs = [str(p) for p in base_pred_responses]

evaluator_dict = {
    "correctness": evaluator_c,
    "faithfulness": evaluator_f,
    "relevancy": evaluator_r,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

eval_results = batch_runner.evaluate_responses(
    eval_qs, responses=pred_responses, reference=ref_response_strs
)

base_eval_results = batch_runner.evaluate_responses(
    eval_qs, responses=base_pred_responses, reference=ref_response_strs
)

results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Auto Merging Retriever", "Base Retriever"],
    ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
)
logger.debug("results_df:")
logger.success(results_df)

# **Analysis**: The results are roughly the same.
#
# Let's also try to see which answer GPT-4 prefers with our pairwise evals.

pairwise_evaluator = PairwiseComparisonEvaluator(llm=eval_llm)
batch_runner = BatchEvalRunner(
    {"pairwise": pairwise_evaluator}, workers=10, show_progress=True
)

pairwise_eval_results = batch_runner.evaluate_response_strs(
    eval_qs,
    response_strs=pred_response_strs,
    reference=base_pred_response_strs,
)
pairwise_score = np.array(
    [r.score for r in pairwise_eval_results["pairwise"]]
).mean()

logger.debug("pairwise_score:")
logger.success(pairwise_score)

# **Analysis**: The pairwise comparison score is a measure of the percentage of time the candidate answer (using auto-merging retriever) is preferred vs. the base answer (using the base retriever). Here we see that it's roughly even.

logger.info("\n\n[DONE]", bright=True)
