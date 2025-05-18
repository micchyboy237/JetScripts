import os
import numpy as np
from typing import List
from helpers import (
    setup_config, load_json_data, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response, evaluate_ai_response, save_file,
    DATA_DIR, DOCS_PATH
)
from jet.logger import CustomLogger

# Setup configuration
script_dir, generated_dir, log_file, logger = setup_config(__file__)

logger.info("Loading JSON data")
formatted_chunks, original_chunks = load_json_data(DOCS_PATH, logger)
# Combine chunks into a single text for sentence splitting
extracted_text = " ".join([chunk.split("\n\n")[-1]
                          for chunk in formatted_chunks])
logger.debug(extracted_text[:500])

# Initialize MLX and embedding function
mlx, embed_func = initialize_mlx(logger)

logger.info("Splitting text into sentences")
sentences = extracted_text.split(". ")
logger.debug(f"Number of sentences: {len(sentences)}")

logger.info("Generating sentence embeddings")
embeddings = generate_embeddings(sentences, embed_func, logger)
logger.debug(f"Generated {len(embeddings)} sentence embeddings.")

logger.info("Calculating cosine similarities")


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


similarities = [cosine_similarity(
    embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

logger.info("Computing breakpoints for semantic chunking")


def compute_breakpoints(similarities, method="percentile", threshold=90):
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


breakpoints = compute_breakpoints(
    similarities, method="percentile", threshold=90)
logger.debug(f"Found {len(breakpoints)} breakpoints")

logger.info("Splitting sentences into semantic chunks")


def split_into_chunks(sentences, breakpoints):
    chunks = []
    start = 0
    for bp in breakpoints:
        chunks.append(". ".join(sentences[start:bp + 1]) + ".")
        start = bp + 1
    chunks.append(". ".join(sentences[start:]))
    return chunks


text_chunks = split_into_chunks(sentences, breakpoints)
logger.debug(f"Number of semantic chunks: {len(text_chunks)}")
logger.debug("\nFirst text chunk:")
logger.debug(text_chunks[0])
save_file({"chunks": text_chunks}, f"{generated_dir}/semantic_chunks.json")
logger.info(f"Saved semantic chunks to {generated_dir}/semantic_chunks.json")

logger.info("Creating embeddings for chunks")
chunk_embeddings = generate_embeddings(text_chunks, embed_func, logger)

logger.info("Performing semantic search")


def semantic_search(query, text_chunks, chunk_embeddings, k=2):
    query_embedding = embed_func(query)
    similarities = [cosine_similarity(query_embedding, emb)
                    for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [{"id": f"chunk_{i}", "rank": idx + 1, "doc_index": i, "score": similarities[i], "text": text_chunks[i]} for idx, i in enumerate(top_indices)]


validation_data = load_validation_data(f"{DATA_DIR}/val.json", logger)
query = validation_data[0]['question']
top_chunks = semantic_search(query, text_chunks, chunk_embeddings, k=2)
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
