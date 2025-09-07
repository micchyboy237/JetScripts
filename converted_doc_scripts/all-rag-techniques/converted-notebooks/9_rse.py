import os
import numpy as np
import json
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response, evaluate_ai_response,
    load_json_data, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def process_document(formatted_texts: List[str], original_chunks: List[Dict[str, Any]]) -> tuple[List[str], Any, Dict[str, Any]]:
    logger.debug("Processing document from preloaded JSON data...")
    logger.debug(f"Loaded {len(formatted_texts)} text chunks")
    logger.debug("Generating embeddings for chunks...")
    chunk_embeddings = generate_embeddings(formatted_texts, embed_func, logger)
    vector_store = SimpleVectorStore()
    metadata = [{"chunk_index": i, "source": DOCS_PATH}
                for i in range(len(formatted_texts))]
    for chunk, embedding, meta in zip(formatted_texts, chunk_embeddings, metadata):
        vector_store.add_item(chunk, embedding, meta)
    doc_info = {
        "chunks": formatted_texts,
        "source": DOCS_PATH,
    }
    return formatted_texts, vector_store, doc_info


def calculate_chunk_values(query: str, chunks: List[str], vector_store: SimpleVectorStore, irrelevant_chunk_penalty: float = 0.2) -> List[float]:
    query_embedding = generate_embeddings(query, embed_func, logger)
    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)
    relevance_scores = {result["metadata"]["chunk_index"]
        : result["similarity"] for result in results}
    chunk_values = []
    for i in range(num_chunks):
        score = relevance_scores.get(i, 0.0)
        value = score - irrelevant_chunk_penalty
        chunk_values.append(value)
    return chunk_values


def find_best_segments(chunk_values: List[float], max_segment_length: int = 20, total_max_length: int = 30, min_segment_value: float = 0.2) -> tuple[List[tuple[int, int]], List[float]]:
    logger.debug("Finding optimal continuous text segments...")
    best_segments = []
    segment_scores = []
    total_included_chunks = 0
    while total_included_chunks < total_max_length:
        best_score = min_segment_value
        best_segment = None
        for start in range(len(chunk_values)):
            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue
            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length
                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue
                segment_value = sum(chunk_values[start:end])
                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)
        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            logger.debug(
                f"Found segment {best_segment} with score {best_score:.4f}")
        else:
            break
    best_segments = sorted(best_segments, key=lambda x: x[0])
    return best_segments, segment_scores


def reconstruct_segments(chunks: List[str], best_segments: List[tuple[int, int]]) -> List[Dict[str, Any]]:
    reconstructed_segments = []
    for start, end in best_segments:
        segment_text = " ".join(chunks[start:end])
        reconstructed_segments.append({
            "text": segment_text,
            "segment_range": (start, end),
        })
    return reconstructed_segments


def format_segments_for_context(segments: List[Dict[str, Any]]) -> str:
    context = []
    for i, segment in enumerate(segments):
        segment_header = f"SEGMENT {i+1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):"
        context.append(segment_header)
        context.append(segment['text'])
        context.append("-" * 80)
    return "\n\n".join(context)


def rag_with_rse(formatted_texts: List[str], original_chunks: List[Dict[str, Any]], query: str, irrelevant_chunk_penalty: float = 0.2) -> Dict[str, Any]:
    logger.debug("\n=== STARTING RAG WITH RELEVANT SEGMENT EXTRACTION ===")
    logger.debug(f"Query: {query}")
    chunks, vector_store, doc_info = process_document(
        formatted_texts, original_chunks)
    logger.debug("\nCalculating relevance scores and chunk values...")
    chunk_values = calculate_chunk_values(
        query, chunks, vector_store, irrelevant_chunk_penalty)
    best_segments, scores = find_best_segments(
        chunk_values,
        max_segment_length=20,
        total_max_length=30,
        min_segment_value=0.2
    )
    logger.debug("\nReconstructing text segments from chunks...")
    segments = reconstruct_segments(chunks, best_segments)
    context = format_segments_for_context(segments)
    system_prompt = (
        "You are a helpful AI assistant. Answer the user's question based only on the provided context. "
        "If you cannot find the answer in the context, state that you don't have enough information."
    )
    response = generate_ai_response(
        query, system_prompt, segments, mlx, logger)
    result = {
        "query": query,
        "segments": segments,
        "response": response
    }
    logger.debug("\n=== FINAL RESPONSE ===")
    logger.debug(response)
    save_file(result, f"{generated_dir}/rse_result.json")
    return result


def standard_top_k_retrieval(formatted_texts: List[str], original_chunks: List[Dict[str, Any]], query: str, k: int = 10) -> Dict[str, Any]:
    logger.debug("\n=== STARTING STANDARD TOP-K RETRIEVAL ===")
    logger.debug(f"Query: {query}")
    chunks, vector_store, doc_info = process_document(
        formatted_texts, original_chunks)
    logger.debug("Creating query embedding and retrieving chunks...")
    query_embedding = generate_embeddings(query, embed_func, logger)
    results = vector_store.search(query_embedding, top_k=k)
    retrieved_chunks = [{"text": result["text"],
                         "metadata": result["metadata"]} for result in results]
    context = "\n\n".join([
        f"CHUNK {i+1}:\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    system_prompt = (
        "You are a helpful AI assistant. Answer the user's question based only on the provided context. "
        "If you cannot find the answer in the context, state that you don't have enough information."
    )
    response = generate_ai_response(
        query, system_prompt, retrieved_chunks, mlx, logger)
    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }
    logger.debug("\n=== FINAL RESPONSE ===")
    logger.debug(response)
    save_file(result, f"{generated_dir}/standard_result.json")
    return result


def evaluate_methods(formatted_texts: List[str], original_chunks: List[Dict[str, Any]], query: str, reference_answer: str = None) -> Dict[str, Any]:
    logger.debug("\n========= EVALUATION =========\n")
    rse_result = rag_with_rse(formatted_texts, original_chunks, query)
    standard_result = standard_top_k_retrieval(
        formatted_texts, original_chunks, query)
    evaluation = {}
    if reference_answer:
        logger.debug("\n=== COMPARING RESULTS ===")
        logger.debug("Evaluating responses against reference answer...")
        evaluation_score, evaluation_text = evaluate_ai_response(
            query, rse_result['response'], reference_answer, mlx, logger)
        evaluation = {
            "rse_evaluation_score": evaluation_score,
            "rse_evaluation_text": evaluation_text
        }
        standard_score, standard_text = evaluate_ai_response(
            query, standard_result['response'], reference_answer, mlx, logger)
        evaluation.update({
            "standard_evaluation_score": standard_score,
            "standard_evaluation_text": standard_text
        })
        logger.debug("\n=== EVALUATION RESULTS ===")
        logger.debug(f"RSE Evaluation Score: {evaluation_score}")
        logger.debug(f"RSE Evaluation Text: {evaluation_text}")
        logger.debug(f"Standard Evaluation Score: {standard_score}")
        logger.debug(f"Standard Evaluation Text: {standard_text}")
    result = {
        "rse_result": rse_result,
        "standard_result": standard_result,
        "evaluation": evaluation
    }
    save_file(result, f"{generated_dir}/evaluation.json")
    return result


script_dir, generated_dir, log_file, logger = setup_config(__file__)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
mlx, embed_func = initialize_mlx(logger)
validation_data = load_validation_data(f"{DATA_DIR}/val.json", logger)
query = validation_data[0]['question']
reference_answer = validation_data[0]['ideal_answer']
results = evaluate_methods(
    formatted_texts, original_chunks, query, reference_answer)
logger.info("\n\n[DONE]", bright=True)
