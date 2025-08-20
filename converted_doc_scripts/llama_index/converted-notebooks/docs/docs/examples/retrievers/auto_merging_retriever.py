import asyncio
from jet.transformers.formatters import format_json
from collections import defaultdict
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import (
CorrectnessEvaluator,
SemanticSimilarityEvaluator,
RelevancyEvaluator,
FaithfulnessEvaluator,
PairwiseComparisonEvaluator,
)
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.core.evaluation.eval_utils import (
get_responses,
get_results_df,
)
from llama_index.core.node_parser import (
HierarchicalNodeParser,
SentenceSplitter,
)
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import asyncio
import numpy as np
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/auto_merging_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Auto Merging Retriever

In this notebook, we showcase our `AutoMergingRetriever`, which looks at a set of leaf nodes and recursively "merges" subsets of leaf nodes that reference a parent node beyond a given threshold. This allows us to consolidate potentially disparate, smaller contexts into a larger context that might help synthesis.

You can define this hierarchy yourself over a set of documents, or you can make use of our brand-new text parser: a HierarchicalNodeParser that takes in a candidate set of documents and outputs an entire hierarchy of nodes, from "coarse-to-fine".
"""
logger.info("# Auto Merging Retriever")

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-file pymupdf

# %load_ext autoreload
# %autoreload 2

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.")

# !pip install llama-index

"""
## Load Data

Let's first load the Llama 2 paper: https://arxiv.org/pdf/2307.09288.pdf. This will be our test data.
"""
logger.info("## Load Data")

# !mkdir -p 'data/'
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O f"{GENERATED_DIR}/llama2.pdf"


# from llama_index.readers.file import PDFReader
# from llama_index.readers.file import PyMuPDFReader

# loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

"""
By default, the PDF reader creates a separate doc for each page.
For the sake of this notebook, we stitch docs together into one doc. 
This will help us better highlight auto-merging capabilities that "stitch" chunks together later on.
"""
logger.info("By default, the PDF reader creates a separate doc for each page.")


doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

"""
## Parse Chunk Hierarchy from Text, Load into Storage

In this section we make use of the `HierarchicalNodeParser`. This will output a hierarchy of nodes, from top-level nodes with bigger chunk sizes to child nodes with smaller chunk sizes, where each child node has a parent node with a bigger chunk size.

By default, the hierarchy is:
- 1st level: chunk size 2048
- 2nd level: chunk size 512
- 3rd level: chunk size 128


We then load these nodes into storage. The leaf nodes are indexed and retrieved via a vector store - these are the nodes that will first be directly retrieved via similarity search. The other nodes will be retrieved from a docstore.
"""
logger.info("## Parse Chunk Hierarchy from Text, Load into Storage")


node_parser = HierarchicalNodeParser.from_defaults()

nodes = node_parser.get_nodes_from_documents(docs)

len(nodes)

"""
Here we import a simple helper function for fetching "leaf" nodes within a node list. 
These are nodes that don't have children of their own.
"""
logger.info("Here we import a simple helper function for fetching "leaf" nodes within a node list.")


leaf_nodes = get_leaf_nodes(nodes)

len(leaf_nodes)

root_nodes = get_root_nodes(nodes)

"""
### Load into Storage

We define a docstore, which we load all nodes into. 

We then define a `VectorStoreIndex` containing just the leaf-level nodes.
"""
logger.info("### Load into Storage")


docstore = SimpleDocumentStore()

docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

llm = MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")


base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)

"""
## Define Retriever
"""
logger.info("## Define Retriever")


base_retriever = base_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

query_str = (
    "What could be the potential outcomes of adjusting the amount of safety"
    " data used in the RLHF stage?"
)

nodes = retriever.retrieve(query_str)
base_nodes = base_retriever.retrieve(query_str)

len(nodes)

len(base_nodes)


for node in nodes:
    display_source_node(node, source_length=10000)

for node in base_nodes:
    display_source_node(node, source_length=10000)

"""
## Plug it into Query Engine
"""
logger.info("## Plug it into Query Engine")


query_engine = RetrieverQueryEngine.from_args(retriever)
base_query_engine = RetrieverQueryEngine.from_args(base_retriever)

response = query_engine.query(query_str)

logger.debug(str(response))

base_response = base_query_engine.query(query_str)

logger.debug(str(base_response))

"""
## Evaluation

We evaluate how well the hierarchical retriever works compared to the baseline retriever in a more quantitative manner.

**WARNING**: This can be *expensive*, especially with GPT-4. Use caution and tune the sample size to fit your budget.
"""
logger.info("## Evaluation")

# import nest_asyncio

# nest_asyncio.apply()

eval_llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
dataset_generator = DatasetGenerator(
    root_nodes[:20],
    llm=eval_llm,
    show_progress=True,
    num_questions_per_chunk=3,
)

async def run_async_code_672f8f1d():
    async def run_async_code_850edf7e():
        eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=60)
        return eval_dataset
    eval_dataset = asyncio.run(run_async_code_850edf7e())
    logger.success(format_json(eval_dataset))
    return eval_dataset
eval_dataset = asyncio.run(run_async_code_672f8f1d())
logger.success(format_json(eval_dataset))

eval_dataset.save_json(f"{GENERATED_DIR}/llama2_eval_qr_dataset.json")

eval_dataset = QueryResponseDataset.from_json(
    f"{GENERATED_DIR}/llama2_eval_qr_dataset.json"
)

"""
### Compare Results

We run evaluations on each of the retrievers: correctness, semantic similarity, relevance, and faithfulness.
"""
logger.info("### Compare Results")

# import nest_asyncio

# nest_asyncio.apply()




evaluator_c = CorrectnessEvaluator(llm=eval_llm)
evaluator_s = SemanticSimilarityEvaluator(llm=eval_llm)
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

async def async_func_51():
    eval_results = batch_runner.evaluate_responses(
        eval_qs, responses=pred_responses, reference=ref_response_strs
    )
    return eval_results
eval_results = asyncio.run(async_func_51())
logger.success(format_json(eval_results))

async def async_func_55():
    base_eval_results = batch_runner.evaluate_responses(
        eval_qs, responses=base_pred_responses, reference=ref_response_strs
    )
    return base_eval_results
base_eval_results = asyncio.run(async_func_55())
logger.success(format_json(base_eval_results))

results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Auto Merging Retriever", "Base Retriever"],
    ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
)
display(results_df)

"""
**Analysis**: The results are roughly the same.

Let's also try to see which answer GPT-4 prefers with our pairwise evals.
"""
logger.info("Let's also try to see which answer GPT-4 prefers with our pairwise evals.")

batch_runner = BatchEvalRunner(
    {"pairwise": pairwise_evaluator}, workers=10, show_progress=True
)

async def async_func_4():
    pairwise_eval_results = batch_runner.evaluate_response_strs(
        eval_qs,
        response_strs=pred_response_strs,
        reference=base_pred_response_strs,
    )
    return pairwise_eval_results
pairwise_eval_results = asyncio.run(async_func_4())
logger.success(format_json(pairwise_eval_results))
pairwise_score = np.array(
    [r.score for r in pairwise_eval_results["pairwise"]]
).mean()

pairwise_score

"""
**Analysis**: The pairwise comparison score is a measure of the percentage of time the candidate answer (using auto-merging retriever) is preferred vs. the base answer (using the base retriever). Here we see that it's roughly even.
"""

logger.info("\n\n[DONE]", bright=True)