import os
import numpy as np
from typing import List, Dict, Any
from helpers import (
    setup_config, load_json_data, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response, evaluate_ai_response, save_file,
    DATA_DIR, DOCS_PATH, SearchResult, SimpleVectorStore, cosine_similarity
)
from jet.logger import CustomLogger


def extract_text_from_chunks(formatted_chunks: List[str], logger: CustomLogger) -> str:
    """Extract text content from formatted chunks, ignoring metadata."""
    extracted_text = " ".join([chunk.split("\n\n")[-1]
                              for chunk in formatted_chunks])
    logger.debug(extracted_text[:500])
    return extracted_text


def split_into_sentences(text: str, logger: CustomLogger) -> List[str]:
    """Split text into sentences, removing trailing periods."""
    sentences = [s.strip().rstrip(".") for s in text.split(". ") if s.strip()]
    logger.debug(f"Number of sentences: {len(sentences)}")
    return sentences


def calculate_cosine_similarities(embeddings: List[np.ndarray], logger: CustomLogger) -> List[float]:
    """Calculate cosine similarities between consecutive sentence embeddings."""
    logger.info("Calculating cosine similarities")
    return [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]


def compute_breakpoints(similarities: List[float], method: str = "percentile", threshold: float = 90) -> List[int]:
    """Compute breakpoints for semantic chunking based on similarity scores."""
    if method == "percentile":
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        q1, q3 = np.percentile(similarities, [25, 75])
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        raise ValueError(
            "Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]


def split_into_semantic_chunks(sentences: List[str], breakpoints: List[int], logger: CustomLogger) -> List[str]:
    """Split sentences into semantic chunks based on breakpoints, ensuring trailing periods."""
    chunks = []
    start = 0
    for bp in breakpoints:
        chunk = ". ".join(sentences[start:bp + 1])
        chunks.append(f"{chunk}.")
        start = bp + 1
    if start < len(sentences):
        chunk = ". ".join(sentences[start:])
        chunks.append(f"{chunk}.")
    logger.debug(f"Number of semantic chunks: {len(chunks)}")
    logger.debug("\nFirst text chunk:")
    logger.debug(chunks[0])
    return chunks


def perform_semantic_search(
    query: str,
    text_chunks: List[str],
    chunk_embeddings: List[np.ndarray],
    embed_func: callable,
    k: int = 2
) -> List[SearchResult]:
    """Perform semantic search on text chunks using query embedding and vector store."""
    vector_store = SimpleVectorStore()
    for i, (text, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
        vector_store.add_item(text, embedding, metadata={}, id=f"chunk_{i}")

    query_embedding = embed_func(query)
    results = vector_store.search(query_embedding, top_k=k)

    return [{
        "id": result["id"],
        "rank": idx + 1,
        "doc_index": int(result["id"].split("_")[1]),
        "score": result["similarity"],
        "text": result["text"],
        "metadata": result["metadata"],
        "relevance_score": None
    } for idx, result in enumerate(results)]


def main():
    """Main function to execute semantic chunking and evaluation pipeline."""
    script_dir, generated_dir, log_file, logger = setup_config(__file__)
    logger.info("Loading JSON data")
    formatted_chunks, original_chunks = load_json_data(DOCS_PATH, logger)
    extracted_text = extract_text_from_chunks(formatted_chunks, logger)
    mlx, embed_func = initialize_mlx(logger)
    logger.info("Splitting text into sentences")
    sentences = split_into_sentences(extracted_text, logger)
    logger.info("Generating sentence embeddings")
    embeddings = generate_embeddings(sentences, embed_func, logger)
    logger.debug(f"Generated {len(embeddings)} sentence embeddings.")
    similarities = calculate_cosine_similarities(embeddings, logger)
    logger.info("Computing breakpoints for semantic chunking")
    breakpoints = compute_breakpoints(
        similarities, method="percentile", threshold=90)
    logger.debug(f"Found {len(breakpoints)} breakpoints")
    text_chunks = split_into_semantic_chunks(sentences, breakpoints, logger)
    save_file({"chunks": text_chunks}, f"{generated_dir}/semantic_chunks.json")
    logger.info(
        f"Saved semantic chunks to {generated_dir}/semantic_chunks.json")
    logger.info("Creating embeddings for chunks")
    chunk_embeddings = generate_embeddings(text_chunks, embed_func, logger)
    logger.info("Performing semantic search")
    validation_data = load_validation_data(f"{DATA_DIR}/val.json", logger)
    query = validation_data[0]['question']
    top_chunks = perform_semantic_search(
        query, text_chunks, chunk_embeddings, embed_func, k=2)
    logger.debug(f"Query: {query}")
    for i, chunk in enumerate(top_chunks):
        logger.debug(f"Context {i+1}:\n{chunk['text']}\n{'='*40}")
    save_file([dict(chunk) for chunk in top_chunks],
              f"{generated_dir}/top_chunks.json")
    logger.info(f"Saved search results to {generated_dir}/top_chunks.json")
    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"
    ai_response = generate_ai_response(
        query, system_prompt, top_chunks, mlx, logger)
    save_file({"question": query, "response": ai_response},
              f"{generated_dir}/ai_response.json")
    logger.info(f"Saved AI response to {generated_dir}/ai_response.json")
    logger.info("Evaluating response")
    true_answer = validation_data[0]['answer']
    evaluation_score, evaluation_text = evaluate_ai_response(
        query, ai_response, true_answer, mlx, logger)
    logger.success(f"Evaluation Score: {evaluation_score}")
    logger.success(f"Evaluation Text: {evaluation_text}")
    save_file({
        "question": query,
        "response": ai_response,
        "true_answer": true_answer,
        "evaluation_score": evaluation_score,
        "evaluation_text": evaluation_text
    }, f"{generated_dir}/evaluation.json")
    logger.info(f"Saved evaluation to {generated_dir}/evaluation.json")
    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
