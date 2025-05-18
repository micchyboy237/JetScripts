import os
import numpy as np
from typing import List, Dict, Any, TypedDict
from jet.file.utils import save_file
from helpers import (
    setup_config, load_json_data, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response, evaluate_ai_response,
    DATA_DIR, DOCS_PATH
)


class SearchResult(TypedDict):
    id: str
    rank: int | None
    doc_index: int
    score: float
    text: str


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def context_enriched_search(
    query: str,
    text_chunks: List[str],
    embeddings: List[np.ndarray],
    chunks: List[Dict[str, Any]],
    embed_func,
    k: int = 1,
    context_size: int = 1
) -> List[SearchResult]:
    """Retrieve top-k chunks with neighboring chunks as SearchResult objects for context enrichment."""
    query_embedding = embed_func(query)
    similarity_scores = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(
            np.array(query_embedding), np.array(chunk_embedding))
        similarity_scores.append((i, similarity_score))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_scores = similarity_scores[:k]
    results = []
    top_index = top_k_scores[0][0]
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size + 1)
    for i in range(start, end):
        score = next((s for idx, s in similarity_scores if idx == i), 0.0)
        rank = 1 if i == top_index else None
        result = SearchResult(
            id=f"chunk_{i}",
            rank=rank,
            doc_index=chunks[i]["metadata"]["doc_index"],
            score=float(score),
            text=text_chunks[i]
        )
        results.append(result)
    return results


# Setup configuration
script_dir, generated_dir, log_file, logger = setup_config(__file__)

# Load docs
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)

# Initialize MLX and embeddings
mlx, embed_func = initialize_mlx(logger)
embeddings = generate_embeddings(formatted_texts, embed_func, logger)

# Load validation data
validation_data = load_validation_data(f"{DATA_DIR}/val.json", logger)
query = validation_data[0]['question']

# Perform context-enriched search
top_chunks = context_enriched_search(
    query, formatted_texts, embeddings, original_chunks, embed_func, k=5, context_size=6
)

# Save search results
save_file([dict(chunk) for chunk in top_chunks],
          f"{generated_dir}/top_chunks.json")
logger.info(f"Saved search results to {generated_dir}/top_chunks.json")

# Generate AI response
system_prompt = (
    "You are a helpful AI Assistant that can read structured and unstructured texts with headers "
    "(lines that start with #). Use the provided context to answer the question accurately."
)
ai_response = generate_ai_response(
    query, system_prompt, top_chunks, mlx, logger)
save_file({"question": query, "response": ai_response},
          f"{generated_dir}/ai_response.json")

# Evaluate response
true_answer = validation_data[0]['answer']
evaluation_score, evaluation_text = evaluate_ai_response(
    query, ai_response, true_answer, mlx, logger)
logger.success(f"Evaluation Score: {evaluation_score}")
logger.success(f"Evaluation Text: {evaluation_text}")

# Save evaluation results
save_file({
    "question": query,
    "response": ai_response,
    "true_answer": true_answer,
    "evaluation_score": evaluation_score,
    "evaluation_text": evaluation_text
}, f"{generated_dir}/evaluation.json")

logger.info("\n\n[DONE]", bright=True)
