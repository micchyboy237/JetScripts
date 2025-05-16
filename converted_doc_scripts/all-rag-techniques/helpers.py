import os
import json
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, TypedDict
from jet.file.utils import load_file, save_file
from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
import re

DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/data/hybrid_reranker_data/anime/top_isekai_anime"
DOCS_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_split_header_docs/searched_html_myanimelist_net_Isekai/chunks.json"


class SearchResult(TypedDict):
    id: str
    rank: int | None
    doc_index: int
    score: float
    text: str


class ValidationData(TypedDict):
    question: str
    answer: str


def setup_config(script_path: str) -> Tuple[str, str, str, CustomLogger]:
    """Set up configuration including directories and logger."""
    script_dir = os.path.dirname(os.path.abspath(script_path))
    log_file = os.path.join(
        script_dir, f"{os.path.splitext(os.path.basename(script_path))[0]}.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    file_name = os.path.splitext(os.path.basename(script_path))[0]
    generated_dir = os.path.join(script_dir, "generated", file_name)
    os.makedirs(generated_dir, exist_ok=True)
    return script_dir, generated_dir, log_file, logger


def load_json_data(data_path: str, logger: CustomLogger) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load pre-chunked data from a JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    formatted_chunks = []
    for chunk in data["chunks"]:
        metadata = chunk["metadata"]
        meta_str = f"[doc_index: {metadata['doc_index']}]\n"
        if metadata['header_level'] != 1:
            meta_str += f"[parent_header: {metadata['parent_header']}]\n"
        meta_str += f"[header: {metadata['header']}]"
        formatted_chunks.append(f"{meta_str.strip()}\n\n{chunk['text']}")
    logger.debug(f"Number of text chunks: {len(formatted_chunks)}")
    logger.debug("\nFirst text chunk:")
    logger.debug(formatted_chunks[0])
    return formatted_chunks, data["chunks"]


def initialize_mlx(logger: CustomLogger) -> Tuple[MLX, Callable]:
    """Initialize MLX model and embedding function."""
    logger.info("Initializing MLX and embedding function")
    mlx = MLX()
    embed_func = get_embedding_function("mxbai-embed-large")
    return mlx, embed_func


def generate_embeddings(texts: List[str], embed_func: Callable, logger: CustomLogger) -> List[np.ndarray]:
    """Generate embeddings for text chunks."""
    logger.info("Generating embeddings for chunks")
    embeddings = embed_func(texts)
    logger.info("Embeddings generated")
    return embeddings


def load_validation_data(val_path: str, logger: CustomLogger) -> List[ValidationData]:
    """Load validation data from JSON file."""
    logger.info("Loading validation data")
    with open(val_path) as f:
        data = json.load(f)
    return data


def generate_ai_response(
    query: str,
    system_prompt: str,
    retrieved_chunks: List[SearchResult],
    mlx: MLX,
    logger: CustomLogger,
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
) -> str:
    """Generate AI response using MLX model."""
    logger.info("Generating AI response")
    context = "\n".join(
        [f"Context {i+1}:\n{chunk['text']}\n=====================================\n" for i, chunk in enumerate(retrieved_chunks)])
    user_prompt = f"{context}\nQuestion: {query}"
    response = mlx.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    ai_response = response["content"]
    logger.success(f"AI Response:\n{ai_response}")
    return ai_response


def evaluate_ai_response(
    question: str,
    response: str,
    true_answer: str,
    mlx: MLX,
    logger: CustomLogger
) -> Tuple[float, str]:
    """Evaluate AI response against true answer."""
    logger.info("Evaluating response")
    evaluate_system_prompt = (
        "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. "
        "If the AI assistant's response is very close to the true response, assign a score of 1. "
        "If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. "
        "If the response is partially aligned with the true response, assign a score of 0.5."
    )

    def parse_score(text: str) -> float:
        match = re.search(r'-?\d*\.?\d+', text)
        if match:
            return float(match.group())
        raise ValueError(f"No valid float found in text: {text}")

    evaluation_prompt = f"User Query: {question}\nAI Response:\n{response}\nTrue Response: {true_answer}\n{evaluate_system_prompt}"
    evaluation_response = mlx.chat([
        {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
        {"role": "user", "content": evaluation_prompt}
    ])
    try:
        score = parse_score(evaluation_response["content"].strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse evaluation score, defaulting to 0")
        score = 0.0
    return score, evaluation_response["content"]
