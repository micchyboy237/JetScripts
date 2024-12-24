import os
from pathlib import Path
import json
import pandas as pd
import nest_asyncio
from llama_index.readers.file import PDFReader
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
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


# Set up the environment
def setup_environment(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    nest_asyncio.apply()


# Download the Llama 2 paper
def download_data():
    os.makedirs("data", exist_ok=True)
    os.system(
        'wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"'
    )


# Load data and create documents
def load_documents(file_path):
    loader = PDFReader()
    docs = loader.load_data(file=Path(file_path))
    doc_text = "\n\n".join([d.get_content() for d in docs])
    return [doc_text]


# Parse nodes from documents
def parse_nodes(docs, chunk_size=1024):
    parser = SentenceSplitter(chunk_size=chunk_size)
    nodes = parser.get_nodes_from_documents(docs)
    for idx, node in enumerate(nodes):
        node.id_ = f"node-{idx}"
    return nodes


# Initialize retrievers and query engines
def initialize_retriever(base_nodes, embed_model_name, similarity_top_k=2):
    embed_model = resolve_embed_model(embed_model_name)
    base_index = VectorStoreIndex(base_nodes, embed_model=embed_model)
    retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
    return retriever


# Perform retrieval and display results
def retrieve_and_display(retriever, query, llm_model):
    llm = OpenAI(model=llm_model)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    response = query_engine.query(query)
    print(str(response))


# Generate metadata references
def generate_metadata(base_nodes, extractors):
    node_to_metadata = {}
    for extractor in extractors:
        metadata_dicts = extractor.extract(base_nodes)
        for node, metadata in zip(base_nodes, metadata_dicts):
            if node.node_id not in node_to_metadata:
                node_to_metadata[node.node_id] = metadata
            else:
                node_to_metadata[node.node_id].update(metadata)
    return node_to_metadata


# Save and load metadata

def save_metadata_dicts(path, data):
    with open(path, "w") as fp:
        json.dump(data, fp)


def load_metadata_dicts(path):
    with open(path, "r") as fp:
        return json.load(fp)


# Generate and evaluate dataset
def evaluate_retrievers(base_nodes, retrievers, eval_dataset_path):
    eval_dataset = EmbeddingQAFinetuneDataset.from_json(eval_dataset_path)
    evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retrievers)
    results = evaluator.aevaluate_dataset(eval_dataset, show_progress=True)
    return results


def display_results(names, results_arr):
    hit_rates, mrrs = [], []
    for name, eval_results in zip(names, results_arr):
        metric_dicts = [eval.metric_vals_dict for eval in eval_results]
        results_df = pd.DataFrame(metric_dicts)
        hit_rates.append(results_df["hit_rate"].mean())
        mrrs.append(results_df["mrr"].mean())
    final_df = pd.DataFrame(
        {"retrievers": names, "hit_rate": hit_rates, "mrr": mrrs})
    print(final_df)


# Main execution pipeline
def main():
    api_key = "YOUR_OPENAI_API_KEY"
    setup_environment(api_key)

    # Data loading
    download_data()
    docs = load_documents("data/llama2.pdf")
    base_nodes = parse_nodes(docs)

    # Initialize retriever
    retriever = initialize_retriever(base_nodes, "local:BAAI/bge-small-en")
    retrieve_and_display(
        retriever, "Can you tell me about the key concepts for safety finetuning", "gpt-3.5-turbo")

    # Metadata generation
    extractors = [SummaryExtractor(summaries=["self"], show_progress=True),
                  QuestionsAnsweredExtractor(questions=5, show_progress=True)]
    metadata = generate_metadata(base_nodes, extractors)
    save_metadata_dicts("data/llama2_metadata.json", metadata)

    # Evaluation
    eval_results = evaluate_retrievers(
        base_nodes, retriever, "data/llama2_eval_dataset.json")
    display_results(["Base Retriever"], [eval_results])


if __name__ == "__main__":
    main()
