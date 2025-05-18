import os
import numpy as np
from typing import List, Dict, Any, TypedDict
from tqdm import tqdm
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response, evaluate_ai_response,
    load_json_data, DATA_DIR, DOCS_PATH
)


class SearchResult(TypedDict):
    id: str
    rank: int | None
    doc_index: int
    score: float
    text: str
    header: str
    embedding: np.ndarray
    header_embedding: np.ndarray


def generate_chunk_header(chunk: str, mlx) -> str:
    """Generate a concise and informative header for a text chunk."""
    system_prompt = "Generate a concise and informative title for the given text."
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )
    return response["content"]


def chunk_text_with_headers(chunks: List[Dict[str, Any]], mlx) -> List[Dict[str, Any]]:
    """Generate headers for pre-chunked text data."""
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        header = generate_chunk_header(text, mlx)
        enhanced_chunks.append({
            "header": header,
            "text": text,
            "doc_index": chunk["metadata"]["doc_index"]
        })
    return enhanced_chunks


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_search(
    query: str,
    chunks: List[Dict[str, Any]],
    embed_func,
    k: int = 2
) -> List[SearchResult]:
    """Perform semantic search using text and header embeddings."""
    query_embedding = embed_func(query)
    similarities = []
    for i, chunk in enumerate(chunks):
        sim_text = cosine_similarity(
            np.array(query_embedding), np.array(chunk["embedding"]))
        sim_header = cosine_similarity(
            np.array(query_embedding), np.array(chunk["header_embedding"]))
        avg_similarity = (sim_text + sim_header) / 2
        result = SearchResult(
            id=f"chunk_{i}",
            rank=1 if avg_similarity == max(
                [s for _, s in similarities] + [avg_similarity]) else None,
            doc_index=chunk["doc_index"],
            score=float(avg_similarity),
            text=chunk["text"],
            header=chunk["header"],
            embedding=chunk["embedding"],
            header_embedding=chunk["header_embedding"]
        )
        similarities.append((result, avg_similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in similarities[:k]]


# Setup configuration and logging
script_dir, generated_dir, log_file, logger = setup_config(__file__)

# Initialize MLX and embedding function
mlx, embed_func = initialize_mlx(logger)

# Load pre-chunked data
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")

# Generate headers for chunks
text_chunks = chunk_text_with_headers(original_chunks, mlx)
logger.debug("Sample Chunk:")
logger.debug(f"Header: {text_chunks[0]['header']}")
logger.debug(f"Content: {text_chunks[0]['text']}")

# Generate embeddings for chunks and headers
embeddings = []
for chunk in tqdm(text_chunks, desc="Generating embeddings"):
    text_embedding = embed_func(chunk["text"])
    header_embedding = embed_func(chunk["header"])
    embeddings.append({
        "header": chunk["header"],
        "text": chunk["text"],
        "doc_index": chunk["doc_index"],
        "embedding": text_embedding,
        "header_embedding": header_embedding
    })
logger.info("Embeddings generated")

# Load validation data
validation_data = load_validation_data(f"{DATA_DIR}/val.json", logger)
query = validation_data[0]['question']

# Perform semantic search
top_chunks = semantic_search(query, embeddings, embed_func, k=2)
logger.debug(f"Query: {query}")
for i, chunk in enumerate(top_chunks):
    logger.debug(f"Header {i+1}: {chunk['header']}")
    logger.debug(f"Content:\n{chunk['text']}\n")

# Save search results
save_file([dict(chunk) for chunk in top_chunks],
          f"{generated_dir}/top_chunks.json")
logger.info(f"Saved search results to {generated_dir}/top_chunks.json")

# Generate AI response
system_prompt = (
    "You are an AI assistant that strictly answers based on the given context. "
    "If the answer cannot be derived directly from the provided context, "
    "respond with: 'I do not have enough information to answer that.'"
)
ai_response = generate_ai_response(
    query, system_prompt, top_chunks, mlx, logger)
save_file({"question": query, "response": ai_response},
          f"{generated_dir}/ai_response.json")
logger.info(f"Saved AI response to {generated_dir}/ai_response.json")

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
logger.info(f"Saved evaluation results to {generated_dir}/evaluation.json")

logger.info("\n\n[DONE]", bright=True)
