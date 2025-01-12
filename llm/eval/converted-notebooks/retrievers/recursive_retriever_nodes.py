import asyncio
import os
import pandas as pd
from jet.llm.ollama.constants import OLLAMA_BASE_EMBED_URL, OLLAMA_LARGE_EMBED_MODEL
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    get_retrieval_results_df,
)
import copy
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
import nest_asyncio
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
import json
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from jet.llm.ollama.base import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from jet.llm.utils import display_jet_source_nodes
from jet.llm.ollama.base import OllamaEmbedding
from jet.cache.joblib import load_from_cache_or_compute
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings, large_embed_model
initialize_ollama_settings()


# Model settings
chunk_size = 512
chunk_overlap = 50

VECTOR_RETRIEVER_SIMILARITY_TOP_K = 3
EVAL_SIMILARITY_TOP_K = 10

EVAL_QA_GENERATE_PROMPT_TMPL = """
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge, generate only questions based on the below query.

You are a Job Interviewer. Your task is to setup {num_questions_per_chunk} questions for an upcoming interview. The questions should assess the applicant's skills, experience, and suitability for the role based on the context provided. Ensure the questions are diverse and relevant to the context.
"""

QUERY = "Can you tell me about yourself?"

CACHE_DIR = f"generated/{os.path.basename(__file__).split(".")[0]}"
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_INDEX_CACHE = os.path.join(CACHE_DIR, "base_index.pkl")
ALL_NODES_AND_VECTOR_INDEX_CHUNK_CACHE = os.path.join(
    CACHE_DIR, "all_nodes_and_vector_index_chunk_cache.pkl")
VECTOR_INDEX_METADATA_CACHE = os.path.join(
    CACHE_DIR, "vector_index_metadata.pkl")

INPUT_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
# INPUT_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/summaries/jet-resume"
GENERATED_DIR = f"generated/{os.path.basename(__file__).split(".")[0]}"
os.makedirs(GENERATED_DIR, exist_ok=True)

# COMBINED_FILE_NAME = "combined.txt"
REQUIRED_EXTS = [
    ".md"
    # ".txt"
]

llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)
eval_llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)
embed_model = OllamaEmbedding(
    model_name=OLLAMA_LARGE_EMBED_MODEL, base_url=OLLAMA_BASE_EMBED_URL)
node_parser = SentenceSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Combine file contents
documents = SimpleDirectoryReader(INPUT_DIR, required_exts=[".md"]).load_data()
texts = [doc.text for doc in documents]

# combined_file_path = os.path.join(INPUT_DIR, COMBINED_FILE_NAME)
# with open(combined_file_path, "w") as f:
#     f.write("\n\n\n".join(texts))

# Read files
docs = SimpleDirectoryReader(
    INPUT_DIR, required_exts=REQUIRED_EXTS).load_data()

base_nodes = node_parser.get_nodes_from_documents(docs)
# Update node ids
# for idx, node in enumerate(base_nodes):
#     node.id_ = f"node-{idx}"


# Example
# Baseline Retriever
#
# Define a baseline retriever that simply fetches the top-k raw text nodes by embedding similarity.

def main_baseline_retriever(
    query=QUERY,
    similarity_top_k=VECTOR_RETRIEVER_SIMILARITY_TOP_K,
):
    base_index = VectorStoreIndex(base_nodes, embed_model=embed_model)
    base_retriever = base_index.as_retriever(
        similarity_top_k=similarity_top_k)

    retrievals = base_retriever.retrieve(
        query
    )

    display_jet_source_nodes(query, retrievals)

    query_engine_base = RetrieverQueryEngine.from_args(base_retriever, llm=llm)

    response = query_engine_base.query(
        query
    )
    logger.success(str(response))

    return base_index


# Example
# Chunk References: Smaller Child Chunks Referring to Bigger Parent Chunk
#
# In this usage example, we show how to build a graph of smaller chunks pointing to bigger parent chunks.
#
# During query-time, we retrieve smaller chunks, but we follow references to bigger chunks. This allows us to have more context for synthesis.

def main_chunk_references_smaller_child_chunks_referring_to_bigger_parent_chunk(
    query=QUERY,
    similarity_top_k=VECTOR_RETRIEVER_SIMILARITY_TOP_K,
):
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

    vector_retriever_chunk = vector_index_chunk.as_retriever(
        similarity_top_k=len(all_nodes))

    retriever_chunk = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
        node_dict=all_nodes_dict,
        verbose=False,
    )

    retrieved_nodes = retriever_chunk.retrieve(query)
    retrieved_nodes_sorted = sorted(
        retrieved_nodes, key=lambda item: item.score, reverse=True)
    display_jet_source_nodes(query, retrieved_nodes_sorted)

    query_engine_chunk = RetrieverQueryEngine.from_args(
        retriever_chunk, llm=llm)

    response = query_engine_chunk.query(
        query
    )
    logger.success(str(response))

    return all_nodes_dict, vector_index_chunk


# Example
# Metadata References: Summaries + Generated Questions referring to a bigger chunk
#
# In this usage example, we show how to define additional context that references the source node.
#
# This additional context includes summaries as well as generated questions.
#
# During query-time, we retrieve smaller chunks, but we follow references to bigger chunks. This allows us to have more context for synthesis.

def main_metadata_reference_summaries_generated_questions(
    query=QUERY,
    similarity_top_k=VECTOR_RETRIEVER_SIMILARITY_TOP_K,
):
    nest_asyncio.apply()

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

    save_metadata_dicts(
        f"{GENERATED_DIR}/llama3_1_metadata_dicts.json", node_to_metadata)

    metadata_dicts = load_metadata_dicts(
        f"{GENERATED_DIR}/llama3_1_metadata_dicts.json")

    all_nodes = copy.deepcopy(base_nodes)
    for node_id, metadata in node_to_metadata.items():
        for val in metadata.values():
            all_nodes.append(IndexNode(text=val, index_id=node_id))

    all_nodes_dict = {n.node_id: n for n in all_nodes}

    vector_index_metadata = VectorStoreIndex(all_nodes)

    vector_retriever_metadata = vector_index_metadata.as_retriever(
        similarity_top_k=similarity_top_k
    )

    retriever_metadata = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_metadata},
        node_dict=all_nodes_dict,
        verbose=False,
    )

    nodes = retriever_metadata.retrieve(
        query
    )

    display_jet_source_nodes(query, nodes)

    query_engine_metadata = RetrieverQueryEngine.from_args(
        retriever_metadata, llm=llm
    )

    response = query_engine_metadata.query(
        query
    )
    logger.success(str(response))

    return vector_index_metadata


# Example
# Evaluation
#
# We evaluate how well our recursive retrieval + node reference methods work. We evaluate both chunk references as well as metadata references. We use embedding similarity lookup to retrieve the reference nodes.
#
# We compare both methods against a baseline retriever where we fetch the raw nodes directly.
#
# In terms of metrics, we evaluate using both hit-rate and MRR.

# Dataset Generation
#
# We first generate a dataset of questions from the set of text chunks.

async def main_evaluation(
    base_index,
    all_nodes_dict,
    vector_index_chunk,
    vector_index_metadata,
    similarity_top_k=EVAL_SIMILARITY_TOP_K, qa_generate_prompt_tmpl=EVAL_QA_GENERATE_PROMPT_TMPL,
):
    # Compare Results
    #
    # We run evaluations on each of the retrievers to measure hit rate and MRR.
    #
    # We find that retrievers with node references (either chunk or metadata) tend to perform better than retrieving the raw chunks.

    nest_asyncio.apply()

    eval_dataset = generate_question_context_pairs(
        nodes=base_nodes,
        llm=eval_llm,
        qa_generate_prompt_tmpl=qa_generate_prompt_tmpl,
    )

    eval_dataset.save_json(f"{GENERATED_DIR}/llama3_2_eval_dataset.json")

    eval_dataset = EmbeddingQAFinetuneDataset.from_json(
        f"{GENERATED_DIR}/llama3_2_eval_dataset.json"
    )

    vector_retriever_chunk = vector_index_chunk.as_retriever(
        similarity_top_k=similarity_top_k
    )
    retriever_chunk = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
        node_dict=all_nodes_dict,
        verbose=False,
    )
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever_chunk
    )
    results_chunk = await retriever_evaluator.aevaluate_dataset(
        eval_dataset, show_progress=True
    )

    vector_retriever_metadata = vector_index_metadata.as_retriever(
        similarity_top_k=similarity_top_k
    )
    retriever_metadata = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_metadata},
        node_dict=all_nodes_dict,
        verbose=False,
    )
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever_metadata
    )
    results_metadata = await retriever_evaluator.aevaluate_dataset(
        eval_dataset, show_progress=True
    )

    base_retriever = base_index.as_retriever(
        similarity_top_k=similarity_top_k)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=base_retriever
    )
    results_base = await retriever_evaluator.aevaluate_dataset(
        eval_dataset, show_progress=True
    )

    names = [
        "Base Retriever",
        "Retriever (Chunk References)",
        "Retriever (Metadata References)",
    ]
    results_arr = [results_base, results_chunk, results_metadata]
    metric_keys = None

    full_results_df = get_retrieval_results_df(
        names=names,
        results_arr=results_arr,
        metric_keys=metric_keys
    )
    # Print the DataFrame as a table
    logger.debug("full_results_df:")
    logger.success(full_results_df.to_string(index=False))


if __name__ == "__main__":
    use_cache = True

    logger.debug("Running main_baseline_retriever...")
    base_index = load_from_cache_or_compute(
        main_baseline_retriever,
        file_path=BASE_INDEX_CACHE,
        use_cache=use_cache,
        similarity_top_k=len(documents),
    )

    logger.debug(
        "Running main_chunk_references_smaller_child_chunks_referring_to_bigger_parent_chunk...")
    all_nodes_dict, vector_index_chunk = load_from_cache_or_compute(
        main_chunk_references_smaller_child_chunks_referring_to_bigger_parent_chunk,
        file_path=ALL_NODES_AND_VECTOR_INDEX_CHUNK_CACHE,
        use_cache=use_cache,
    )

    logger.debug(
        "Running main_metadata_reference_summaries_generated_questions...")
    vector_index_metadata = load_from_cache_or_compute(
        main_metadata_reference_summaries_generated_questions,
        file_path=VECTOR_INDEX_METADATA_CACHE,
        use_cache=use_cache,
    )

    logger.debug("Running main_evaluation...")
    asyncio.run(main_evaluation(
        base_index,
        all_nodes_dict,
        vector_index_chunk,
        vector_index_metadata,
    ))
    logger.info("\n\n[DONE]", bright=True)
