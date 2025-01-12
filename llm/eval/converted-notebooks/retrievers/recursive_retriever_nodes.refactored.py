import asyncio
import os
from pathlib import Path
import json
import pandas as pd
from jet.llm.ollama.base import Ollama, initialize_ollama_settings
from jet.logger import logger
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from jet.llm.utils import display_jet_source_node
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    get_retrieval_results_df,
)
llm_settings = initialize_ollama_settings()

GENERATED_DIR = os.path.join(
    "generated/" + os.path.splitext(os.path.basename(__file__))[0])
OUTPUT_DIR = F"{GENERATED_DIR}/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download the Llama 2 paper


def download_data(data_path):
    if not os.path.exists(data_path):
        os.makedirs("data", exist_ok=True)
        os.system(
            f'wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "{
                data_path}"'
        )
    else:
        logger.success("Loaded cached file:", data_path)


# Load data and create documents
def load_documents(file_path):
    loader = SimpleDirectoryReader(input_files=[file_path])
    docs = loader.load_data(num_workers=4, show_progress=True)
    return docs


# Parse nodes from documents
def parse_nodes(docs, chunk_size=1024, chunk_overlap=200):
    parser = SentenceSplitter(chunk_size=chunk_size,
                              chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(docs, show_progress=True)
    for idx, node in enumerate(nodes):
        node.id_ = f"node-{idx}"
    return nodes


# Initialize retrievers and query engines
def initialize_retriever(base_nodes, embed_model_name, similarity_top_k=2):
    embed_model = resolve_embed_model(embed_model_name)
    base_index = VectorStoreIndex(
        base_nodes, embed_model=embed_model, use_async=False, show_progress=True)
    retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
    return retriever


# Perform retrieval and display results
def get_query_engine(retriever, llm_model="llama3.1"):
    llm = Ollama(model=llm_model)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    return query_engine


# Generate metadata references
async def generate_metadata(base_nodes, extractors: list[BaseExtractor]):
    node_to_metadata = {}
    for extractor in extractors:
        metadata_dicts = await extractor.aextract(base_nodes)
        for node, metadata in zip(base_nodes, metadata_dicts):
            if node.node_id not in node_to_metadata:
                node_to_metadata[node.node_id] = metadata
            else:
                node_to_metadata[node.node_id].update(metadata)
    return node_to_metadata


# Save and load metadata

def save_metadata_dicts(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(make_serializable(data), fp, indent=2, ensure_ascii=False)
    logger.success("Saved file to" + path, bright=True)


def load_metadata_dicts(path):
    logger.debug(f"Loading file from: {path}")
    with open(path, "r") as fp:
        return json.load(fp)


# Generate and evaluate dataset
async def evaluate_retrievers(base_nodes, retrievers, eval_dataset_path):
    eval_dataset = EmbeddingQAFinetuneDataset.from_json(eval_dataset_path)
    evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retrievers)
    results = await evaluator.aevaluate_dataset(eval_dataset, show_progress=True)
    return results


def display_results(names, results_arr):
    results_list = []
    for name, eval_results in zip(names, results_arr):
        metric_dicts = [eval.metric_vals_dict for eval in eval_results]
        hit_rate = sum(d["hit_rate"] for d in metric_dicts) / len(metric_dicts)
        mrr = sum(d["mrr"] for d in metric_dicts) / len(metric_dicts)
        results_list.append(
            {"retriever": name, "hit_rate": hit_rate, "mrr": mrr})
    return results_list


# Main execution pipeline
async def main():
    chunk_size = 1024
    chunk_overlap = 200
    top_k = 2
    query = "Can you tell me about the key concepts for safety finetuning"

    # Data loading
    logger.newline()
    logger.info("Data loading...")
    download_data(f"{GENERATED_DIR}/llama2.pdf")
    docs = load_documents(f"{GENERATED_DIR}/llama2.pdf")
    logger.log("All docs:", len(docs), colors=["DEBUG", "SUCCESS"])
    base_nodes = parse_nodes(docs, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap)
    logger.log("Parsed nodes:", len(base_nodes), colors=["DEBUG", "SUCCESS"])

    # Initialize retriever
    logger.newline()
    logger.info("Initialize retriever...")
    retriever = initialize_retriever(
        base_nodes, "local:BAAI/bge-small-en", similarity_top_k=top_k)

    logger.newline()
    logger.info("Querying retriever...")
    logger.debug(query)
    query_engine = get_query_engine(retriever)
    response = query_engine.query(query)
    logger.success(format_json(response))

    # Metadata generation
    logger.info("Metadata generation...")
    summary_extractor = SummaryExtractor(llm=llm_settings.llm, summaries=[
        "self"], show_progress=True)
    questions_answered_extractor = QuestionsAnsweredExtractor(
        llm=llm_settings.llm, questions=5, show_progress=True)
    extractors = [summary_extractor, questions_answered_extractor]
    metadata = await generate_metadata(base_nodes, extractors)
    save_metadata_dicts(f"{OUTPUT_DIR}/llama2_metadata.json", metadata)

    logger.newline()
    logger.debug("Extracted metadata results:")
    logger.success(format_json(metadata))

    # Evaluation
    logger.info("Evaluation...")
    eval_results = await evaluate_retrievers(
        base_nodes, retriever, f"{OUTPUT_DIR}/llama2_eval_dataset.json")
    display_results(["Base Retriever"], [eval_results])

    logger.newline()
    logger.debug("Eval results:")
    logger.success(format_json(eval_results))


if __name__ == "__main__":
    asyncio.run(main())
