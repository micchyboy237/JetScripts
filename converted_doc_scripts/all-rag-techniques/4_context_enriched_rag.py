from typing import List, Union, Literal, Callable
from typing import List
from typing import List, Dict, Any
import re
from jet.file.utils import load_file, save_file
from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
from tqdm import tqdm
import pypdf
import json
import numpy as np
import os
from typing import TypedDict, Optional, List, Tuple, Dict, Any, Callable


class SearchResult(TypedDict):
    id: str
    rank: Optional[int]
    doc_index: int
    score: float
    text: str


class ValidationData(TypedDict):
    question: str
    answer: str


script_dir: str = os.path.dirname(os.path.abspath(__file__))
log_file: str = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger: CustomLogger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")
file_name: str = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR: str = os.path.join(script_dir, "generated", file_name)
DATA_DIR: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/data/hybrid_reranker_data/anime/top_isekai_anime"
os.makedirs(GENERATED_DIR, exist_ok=True)


def load_data_from_json(data_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load pre-chunked data from a JSON file and return text with metadata and original chunks.

    Args:
        data_path (str): Path to the JSON file containing chunked data.

    Returns:
        Tuple[List[str], List[Dict[str, Any]]]: List of formatted text strings and list of original chunk dictionaries.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    formatted_chunks = []
    for chunk in data["chunks"]:
        metadata = chunk["metadata"]
        meta_str = (
            f"[doc_index: {metadata['doc_index']}]\n"
        )
        if metadata['header_level'] != 1:
            meta_str += (
                f"[parent_header: {metadata['parent_header']}]\n"
            )
        meta_str += f"[header: {metadata['header']}]"
        formatted_chunks.append(f"{meta_str.strip()}\n\n{chunk['text']}")
    return formatted_chunks, data["chunks"]


data_path: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_split_header_docs/searched_html_myanimelist_net_Isekai/chunks.json"
text_chunks: List[Dict[str, Any]] = load_data_from_json(data_path)
logger.debug(f"Number of text chunks: {len(text_chunks)}")
logger.debug("\nFirst text chunk:")
logger.debug(text_chunks[0])
logger.info("Initializing MLX and embedding function")
mlx: MLX = MLX()
embed_func: Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]] = get_embedding_function(
    "mxbai-embed-large")

logger.info("Generating embeddings for chunks")


def create_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Convert text chunks into numerical embeddings using the embedding function."""
    return embed_func(texts)


formatted_texts, original_chunks = load_data_from_json(data_path)
logger.debug(f"Number of text chunks: {len(formatted_texts)}")
logger.debug("\nFirst text chunk:")
logger.debug(formatted_texts[0])

response: List[np.ndarray] = create_embeddings(formatted_texts)


logger.info("Embeddings generated")

logger.info("Defining similarity and context-enriched search functions")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def context_enriched_search(
    query: str,
    text_chunks: List[str],
    embeddings: List[np.ndarray],
    chunks: List[Dict[str, Any]],
    k: int = 1,
    context_size: int = 1
) -> List[SearchResult]:
    """Retrieve top-k chunks with neighboring chunks as SearchResult objects for context enrichment."""
    query_embedding: np.ndarray = embed_func(query)
    similarity_scores: List[Tuple[int, float]] = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score: float = cosine_similarity(
            np.array(query_embedding), np.array(chunk_embedding))
        similarity_scores.append((i, similarity_score))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_scores: List[Tuple[int, float]] = similarity_scores[:k]
    results: List[SearchResult] = []
    top_index: int = top_k_scores[0][0]
    start: int = max(0, top_index - context_size)
    end: int = min(len(text_chunks), top_index + context_size + 1)
    for i in range(start, end):
        score: float = next(
            (s for idx, s in similarity_scores if idx == i), 0.0)
        rank: Optional[int] = 1 if i == top_index else None
        result: SearchResult = SearchResult(
            id=f"chunk_{i}",
            rank=rank,
            doc_index=chunks[i]["metadata"]["doc_index"],
            score=float(score),
            text=text_chunks[i]
        )
        results.append(result)
    return results


logger.info("Loading validation data")
with open(f"{DATA_DIR}/val.json") as f:
    data: List[ValidationData] = json.load(f)
query: str = data[0]['question']
top_chunks: List[SearchResult] = context_enriched_search(
    query, formatted_texts, response, original_chunks, k=5, context_size=6)

logger.debug(f"Query: {query}")
for i, chunk in enumerate(top_chunks):
    logger.debug(
        f"Context {i + 1}:\n{chunk['text']}\nScore: {chunk['score']}\nRank: {chunk['rank']}\n=====================================")
save_file([dict(chunk) for chunk in top_chunks],
          f"{GENERATED_DIR}/top_chunks.json")
logger.info(f"Saved search results to {GENERATED_DIR}/top_chunks.json")

logger.info("Generating AI response")
system_prompt: str = "You are a helpful AI Assistant that can read structured and unstructured texts with headers (lines that start with #). Use the provided context to answer the question accurately."


def generate_response(
    query: str,
    system_prompt: str,
    retrieved_chunks: List[SearchResult],
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
) -> str:
    """Generate a response using the MLX model with retrieved context and query."""
    context: str = "\n".join(
        [f"Context {i+1}:\n{chunk['text']}\n=====================================\n" for i, chunk in enumerate(retrieved_chunks)])
    user_prompt: str = f"{context}\nQuestion: {query}"
    response: Dict[str, Any] = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response["content"]


ai_response: str = generate_response(query, system_prompt, top_chunks)
logger.success(f"AI Response:\n{ai_response}")
save_file({"question": query, "response": ai_response},
          f"{GENERATED_DIR}/ai_response.json")


logger.info("Evaluating response")
evaluate_system_prompt: str = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."


def evaluate_response(question: str, response: str, true_answer: str) -> float:
    """Evaluate the AI response by comparing it to the true answer and assigning a score."""
    def parse_score(text: str) -> float:
        """Parse a numerical score from the evaluation response."""
        match = re.search(r'-?\d*\.?\d+', text)
        if match:
            return float(match.group())
        raise ValueError(f"No valid float found in text: {text}")
    evaluation_prompt: str = f"User Query: {question}\nAI Response:\n{response}\nTrue Response: {true_answer}\n{evaluate_system_prompt}"
    evaluation_response: Dict[str, Any] = mlx.chat(
        [
            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
            {"role": "user", "content": evaluation_prompt}
        ]
    )
    try:
        score: float = parse_score(evaluation_response["content"].strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse evaluation score, defaulting to 0")
        score = 0.0
    return score, evaluation_response["content"]


true_answer: str = data[0]['answer']


# After evaluation
evaluation_score, evaluation_text = evaluate_response(
    query, ai_response, true_answer)
logger.success(f"Evaluation Score: {evaluation_score}")
logger.success(f"Evaluation Text: {evaluation_text}")

# Save evaluation results
save_file({
    "question": query,
    "response": ai_response,
    "true_answer": true_answer,
    "evaluation_score": evaluation_score,
    "evaluation_text": evaluation_text
}, f"{GENERATED_DIR}/evaluation.json")


logger.info("\n\n[DONE]", bright=True)
