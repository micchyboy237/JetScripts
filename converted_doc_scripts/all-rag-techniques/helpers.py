import os
import json
from jet.code.markdown_types.markdown_parsed_types import HeaderSearchResult
from jet.models.embeddings.base import get_embedding_function
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, TypedDict
from jet.file.utils import load_file, save_file
from jet.llm.mlx.base import MLX
from jet.models.model_types import LLMModelType
from jet.logger import CustomLogger
import re

DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/data/hybrid_reranker_data/anime/top_isekai_anime"
DOCS_PATH = f"{DATA_DIR}/search_results.json"
LLM_MODEL = "llama-3.2-3b-instruct-4bit"


class SearchResult(TypedDict):
    id: str
    rank: int | None
    doc_index: int
    score: float
    text: str
    metadata: Dict[str, Any]
    relevance_score: float | None


class ValidationData(TypedDict):
    question: str
    answer: str


class ChunkMetadata(TypedDict):
    doc_index: int
    header: str
    parent_header: str
    header_level: int


class Chunk(TypedDict):
    metadata: ChunkMetadata
    text: str


class ChunkData(TypedDict):
    chunks: List[Chunk]


class VectorStoreItem(TypedDict, total=False):
    id: str
    text: str
    metadata: Dict[str, Any]
    similarity: float


def setup_config(script_path: str) -> Tuple[str, str, str, CustomLogger]:
    script_dir = os.path.dirname(os.path.abspath(script_path))
    log_file = os.path.join(
        script_dir, f"{os.path.splitext(os.path.basename(script_path))[0]}.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    file_name = os.path.splitext(os.path.basename(script_path))[0]
    generated_dir = os.path.join(script_dir, "generated", file_name)
    os.makedirs(generated_dir, exist_ok=True)
    return script_dir, generated_dir, log_file, logger


def load_json_data(data_path: str, logger: CustomLogger) -> Tuple[List[str], List[HeaderSearchResult]]:
    with open(data_path, 'r') as f:
        data: List[HeaderSearchResult] = json.load(f)["results"]
    formatted_chunks = []
    for chunk in data:
        text = f"{chunk['header']}\n{chunk['content']}"
        chunk['text'] = text
        metadata = chunk["metadata"]
        meta_str = f"[doc_index: {metadata['doc_index']}]\n"
        # Add chunk_idx to meta_str
        meta_str += f"[chunk_idx: {metadata['chunk_idx']}]\n"
        # Use parent_header if header_level is not 1
        if metadata.get('level', 1) != 1:
            meta_str += f"[parent_header: {chunk.get('parent_header', '')}]\n"
        meta_str += f"[header: {chunk['header']}]"

        formatted_chunks.append(f"{meta_str.strip()}\n\n{text}")
    logger.debug(f"Number of text chunks: {len(formatted_chunks)}")
    logger.debug("\nFirst text chunk:")
    logger.debug(formatted_chunks[0])
    return formatted_chunks, data


def initialize_mlx(logger: CustomLogger) -> Tuple[MLX, Callable[[str | List[str]], List[float] | List[List[float]]]]:
    logger.info("Initializing MLX and embedding function")
    mlx = MLX()
    embed_func = get_embedding_function("mxbai-embed-large")
    return mlx, embed_func


def generate_embeddings(
    texts: str | List[str],
    embed_func: Callable[[str | List[str]], List[float] | List[List[float]]],
    logger: CustomLogger
) -> np.ndarray | List[np.ndarray]:
    logger.info("Generating embeddings for chunks")
    embeddings = embed_func(texts)
    logger.info("Embeddings generated")
    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


def load_validation_data(val_path: str, logger: CustomLogger) -> List[ValidationData]:
    logger.info("Loading validation data")
    with open(val_path) as f:
        data: List[ValidationData] = json.load(f)
    return data


def generate_ai_response(
    query: str,
    system_prompt: str,
    retrieved_chunks: List[SearchResult],
    mlx: MLX,
    logger: CustomLogger,
    model: LLMModelType = LLM_MODEL,
    **kwargs
) -> str:
    logger.info(f"Generating AI response for model: {model}")
    context = "\n".join([
        f"Context {i+1}:\n{chunk['text']}\n=====================================\n"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    user_prompt = f"{context}\nQuestion: {query}"
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        **kwargs
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    logger.success(f"AI Response:\n{response}")
    return response


def evaluate_ai_response(
    question: str,
    response: str,
    true_answer: str,
    mlx: MLX,
    logger: CustomLogger,
    model: LLMModelType = LLM_MODEL,
    **kwargs
) -> Tuple[float, str]:
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
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
            {"role": "user", "content": evaluation_prompt}
        ],
        model=model,
        **kwargs
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    try:
        score = parse_score(response.strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse evaluation score, defaulting to 0")
        score = 0.0
    return score, response


class SimpleVectorStore:
    def __init__(self) -> None:
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []

    def add_item(self, text: str, embedding: List[float] | np.ndarray, metadata: Dict[str, Any] | None = None, id: str | None = None) -> None:
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
        self.ids.append(id if id is not None else str(len(self.ids)))

    def search(self, query_embedding: List[float] | np.ndarray, top_k: int = 5) -> List[VectorStoreItem]:
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding).flatten()
        similarities: List[Tuple[int, float]] = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
            dot_product = np.dot(query_vector, vector)
            query_norm = np.linalg.norm(query_vector)
            vector_norm = np.linalg.norm(vector)
            similarity = 0.0 if query_norm == 0 or vector_norm == 0 else dot_product / \
                (query_norm * vector_norm)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: -float('inf')
                          if np.isnan(x[1]) else x[1], reverse=True)
        results: List[VectorStoreItem] = []
        for i in range(min(top_k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "id": self.ids[idx],
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        return results
