from jet.transformers.formatters import format_json
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    get_retrieval_results_df,
)
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from pathlib import Path
import copy
import json
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/recursive_retriever_nodes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Recursive Retriever + Node References

This guide shows how you can use recursive retrieval to traverse node relationships and fetch nodes based on "references".

Node references are a powerful concept. When you first perform retrieval, you may want to retrieve the reference as opposed to the raw text. You can have multiple references point to the same node.

In this guide we explore some different usages of node references:
- **Chunk references**: Different chunk sizes referring to a bigger chunk
- **Metadata references**: Summaries + Generated Questions referring to a bigger chunk
"""
logger.info("# Recursive Retriever + Node References")

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-file

# %load_ext autoreload
# %autoreload 2
# %env OPENAI_API_KEY=YOUR_OPENAI_KEY

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info(
    "If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.")

# !pip install llama-index pypdf

"""
## Load Data + Setup

In this section we download the Llama 2 paper and create an initial set of nodes (chunk size 1024).
"""
logger.info("## Load Data + Setup")

# !mkdir -p 'data/'
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

# from llama_index.readers.file import PDFReader

# loader = PDFReader()
docs0 = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]


node_parser = SentenceSplitter(chunk_size=1024)

base_nodes = node_parser.get_nodes_from_documents(docs)
for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"


embed_model = resolve_embed_model("local:BAAI/bge-small-en")
llm = OllamaFunctionCalling(model="llama3.2")

"""
## Baseline Retriever

Define a baseline retriever that simply fetches the top-k raw text nodes by embedding similarity.
"""
logger.info("## Baseline Retriever")

base_index = VectorStoreIndex(base_nodes, embed_model=embed_model)
base_retriever = base_index.as_retriever(similarity_top_k=2)

retrievals = base_retriever.retrieve(
    "Can you tell me about the key concepts for safety finetuning"
)

for n in retrievals:
    display_source_node(n, source_length=1500)

query_engine_base = RetrieverQueryEngine.from_args(base_retriever, llm=llm)

response = query_engine_base.query(
    "Can you tell me about the key concepts for safety finetuning"
)
logger.debug(str(response))

"""
## Chunk References: Smaller Child Chunks Referring to Bigger Parent Chunk

In this usage example, we show how to build a graph of smaller chunks pointing to bigger parent chunks.

During query-time, we retrieve smaller chunks, but we follow references to bigger chunks. This allows us to have more context for synthesis.
"""
logger.info(
    "## Chunk References: Smaller Child Chunks Referring to Bigger Parent Chunk")

sub_chunk_sizes = [128, 256, 512]
sub_node_parsers = [
    SentenceSplitter(chunk_size=c, chunk_overlap=20) for c in sub_chunk_sizes
]

all_nodes = []
for base_node in base_nodes:
    for n in sub_node_parsers:
        sub_nodes = n.get_nodes_from_documents([base_node])
        sub_inodes = [
            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
        ]
        all_nodes.extend(sub_inodes)

    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)

all_nodes_dict = {n.node_id: n for n in all_nodes}

vector_index_chunk = VectorStoreIndex(all_nodes, embed_model=embed_model)

vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=2)

retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)

nodes = retriever_chunk.retrieve(
    "Can you tell me about the key concepts for safety finetuning"
)
for node in nodes:
    display_source_node(node, source_length=2000)

query_engine_chunk = RetrieverQueryEngine.from_args(retriever_chunk, llm=llm)

response = query_engine_chunk.query(
    "Can you tell me about the key concepts for safety finetuning"
)
logger.debug(str(response))

"""
## Metadata References: Summaries + Generated Questions referring to a bigger chunk

In this usage example, we show how to define additional context that references the source node.

This additional context includes summaries as well as generated questions.

During query-time, we retrieve smaller chunks, but we follow references to bigger chunks. This allows us to have more context for synthesis.
"""
logger.info(
    "## Metadata References: Summaries + Generated Questions referring to a bigger chunk")

# import nest_asyncio

# nest_asyncio.apply()


extractors = [
    SummaryExtractor(summaries=["self"], show_progress=True),
    QuestionsAnsweredExtractor(questions=5, show_progress=True),
]

node_to_metadata = {}
for extractor in extractors:
    metadata_dicts = extractor.extract(base_nodes)
    for node, metadata in zip(base_nodes, metadata_dicts):
        if node.node_id not in node_to_metadata:
            node_to_metadata[node.node_id] = metadata
        else:
            node_to_metadata[node.node_id].update(metadata)


def save_metadata_dicts(path, data):
    with open(path, "w") as fp:
        json.dump(data, fp)


def load_metadata_dicts(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


save_metadata_dicts("data/llama2_metadata_dicts.json", node_to_metadata)

metadata_dicts = load_metadata_dicts("data/llama2_metadata_dicts.json")


all_nodes = copy.deepcopy(base_nodes)
for node_id, metadata in node_to_metadata.items():
    for val in metadata.values():
        all_nodes.append(IndexNode(text=val, index_id=node_id))

all_nodes_dict = {n.node_id: n for n in all_nodes}


llm = OllamaFunctionCalling(model="llama3.2")

vector_index_metadata = VectorStoreIndex(all_nodes)

vector_retriever_metadata = vector_index_metadata.as_retriever(
    similarity_top_k=2
)

retriever_metadata = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_metadata},
    node_dict=all_nodes_dict,
    verbose=False,
)

nodes = retriever_metadata.retrieve(
    "Can you tell me about the key concepts for safety finetuning"
)
for node in nodes:
    display_source_node(node, source_length=2000)

query_engine_metadata = RetrieverQueryEngine.from_args(
    retriever_metadata, llm=llm
)

response = query_engine_metadata.query(
    "Can you tell me about the key concepts for safety finetuning"
)
logger.debug(str(response))

"""
## Evaluation

We evaluate how well our recursive retrieval + node reference methods work. We evaluate both chunk references as well as metadata references. We use embedding similarity lookup to retrieve the reference nodes.

We compare both methods against a baseline retriever where we fetch the raw nodes directly.

In terms of metrics, we evaluate using both hit-rate and MRR.

### Dataset Generation

We first generate a dataset of questions from the set of text chunks.
"""
logger.info("## Evaluation")


# import nest_asyncio

# nest_asyncio.apply()

eval_dataset = generate_question_context_pairs(
    base_nodes, OllamaFunctionCalling(model="llama3.2")
)

eval_dataset.save_json("data/llama2_eval_dataset.json")

eval_dataset = EmbeddingQAFinetuneDataset.from_json(
    "data/llama2_eval_dataset.json"
)

"""
### Compare Results

We run evaluations on each of the retrievers to measure hit rate and MRR.

We find that retrievers with node references (either chunk or metadata) tend to perform better than retrieving the raw chunks.
"""
logger.info("### Compare Results")


top_k = 10


def display_results(names, results_arr):
    """Display results from evaluate."""

    hit_rates = []
    mrrs = []
    for name, eval_results in zip(names, results_arr):
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)
        results_df = pd.DataFrame(metric_dicts)

        hit_rate = results_df["hit_rate"].mean()
        mrr = results_df["mrr"].mean()
        hit_rates.append(hit_rate)
        mrrs.append(mrr)

    final_df = pd.DataFrame(
        {"retrievers": names, "hit_rate": hit_rates, "mrr": mrrs}
    )
    display(final_df)


vector_retriever_chunk = vector_index_chunk.as_retriever(
    similarity_top_k=top_k
)
retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever_chunk
)
results_chunk = retriever_evaluator.evaluate_dataset(
    eval_dataset, show_progress=True
)
logger.success(format_json(results_chunk))

vector_retriever_metadata = vector_index_metadata.as_retriever(
    similarity_top_k=top_k
)
retriever_metadata = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_metadata},
    node_dict=all_nodes_dict,
    verbose=True,
)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever_metadata
)
results_metadata = retriever_evaluator.evaluate_dataset(
    eval_dataset, show_progress=True
)
logger.success(format_json(results_metadata))

base_retriever = base_index.as_retriever(similarity_top_k=top_k)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=base_retriever
)
results_base = retriever_evaluator.evaluate_dataset(
    eval_dataset, show_progress=True
)
logger.success(format_json(results_base))

full_results_df = get_retrieval_results_df(
    [
        "Base Retriever",
        "Retriever (Chunk References)",
        "Retriever (Metadata References)",
    ],
    [results_base, results_chunk, results_metadata],
)
display(full_results_df)

logger.info("\n\n[DONE]", bright=True)
