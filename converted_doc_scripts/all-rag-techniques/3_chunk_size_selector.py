import os
import numpy as np
from tqdm import tqdm
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
# Combine chunks into a single text for chunking
extracted_text = " ".join([chunk.split("\n\n")[-1]
                          for chunk in formatted_chunks])
logger.debug(extracted_text[:500])

# Initialize MLX and embedding function
mlx, embed_func = initialize_mlx(logger)

logger.info("Chunking text with different sizes")


def chunk_text(text, n, overlap):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks


chunk_sizes = [128, 256, 512]
text_chunks_dict = {size: chunk_text(
    extracted_text, size, size // 5) for size in chunk_sizes}
for size, chunks in text_chunks_dict.items():
    logger.debug(f"Chunk Size: {size}, Number of Chunks: {len(chunks)}")
save_file({"chunk_sizes": text_chunks_dict},
          f"{generated_dir}/text_chunks.json")
logger.info(f"Saved text chunks to {generated_dir}/text_chunks.json")

logger.info("Generating embeddings for chunks")
chunk_embeddings_dict = {
    size: generate_embeddings(chunks, embed_func, logger)
    for size, chunks in tqdm(text_chunks_dict.items(), desc="Generating Embeddings")
}


logger.info("Defining similarity and retrieval functions")


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve_relevant_chunks(query, text_chunks, chunk_embeddings, k=5):
    query_embedding = embed_func(query)
    similarities = [cosine_similarity(query_embedding, emb)
                    for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [{"id": f"chunk_{i}", "rank": idx + 1, "doc_index": i, "score": similarities[i], "text": text_chunks[i]} for idx, i in enumerate(top_indices)]


validation_data = load_validation_data(f"{DATA_DIR}/val.json", logger)
query = validation_data[3]['question']
retrieved_chunks_dict = {size: retrieve_relevant_chunks(
    query, text_chunks_dict[size], chunk_embeddings_dict[size]) for size in chunk_sizes}
logger.debug(retrieved_chunks_dict[256])
save_file(retrieved_chunks_dict, f"{generated_dir}/retrieved_chunks.json")
logger.info(f"Saved retrieved chunks to {generated_dir}/retrieved_chunks.json")

logger.info("Generating AI responses")
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"
ai_responses_dict = {size: generate_ai_response(
    query, system_prompt, retrieved_chunks_dict[size], mlx, logger) for size in chunk_sizes}
logger.debug(ai_responses_dict[256])
save_file(ai_responses_dict, f"{generated_dir}/ai_responses.json")
logger.info(f"Saved AI responses to {generated_dir}/ai_responses.json")

logger.info("Evaluating responses")
SCORE_FULL = 1.0
SCORE_PARTIAL = 0.5
SCORE_NONE = 0.0

FAITHFULNESS_PROMPT_TEMPLATE = """
Evaluate the faithfulness of the AI response compared to the true answer.
User Query: {question}
AI Response: {response}
True Answer: {true_answer}

Faithfulness measures how well the AI response aligns with facts in the true answer, without hallucinations.

INSTRUCTIONS:
- Score STRICTLY using only these values:
    * {full} = Completely faithful, no contradictions with true answer
    * {partial} = Partially faithful, minor contradictions
    * {none} = Not faithful, major contradictions or hallucinations
- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
"""

RELEVANCY_PROMPT_TEMPLATE = """
Evaluate the relevancy of the AI response to the user query.
User Query: {question}
AI Response: {response}

Relevancy measures how well the response addresses the user's question.

INSTRUCTIONS:
- Score STRICTLY using only these values:
    * {full} = Completely relevant, directly addresses the query
    * {partial} = Partially relevant, addresses some aspects
    * {none} = Not relevant, fails to address the query
- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
"""


def evaluate_response(question, response, true_answer):
    faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
        true_answer=true_answer,
        full=SCORE_FULL,
        partial=SCORE_PARTIAL,
        none=SCORE_NONE
    )
    relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
        full=SCORE_FULL,
        partial=SCORE_PARTIAL,
        none=SCORE_NONE
    )
    faithfulness_response = mlx.chat(
        [
            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
            {"role": "user", "content": faithfulness_prompt}
        ]
    )
    relevancy_response = mlx.chat(
        [
            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
            {"role": "user", "content": relevancy_prompt}
        ]
    )
    try:
        faithfulness_score = float(faithfulness_response["content"].strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse faithfulness score, defaulting to 0")
        faithfulness_score = 0.0
    try:
        relevancy_score = float(relevancy_response["content"].strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse relevancy score, defaulting to 0")
        relevancy_score = 0.0
    return faithfulness_score, relevancy_score


true_answer = validation_data[3]['answer']
faithfulness2, relevancy2 = evaluate_response(
    query, ai_responses_dict[256], true_answer)
logger.debug(f"\n")
logger.debug(f"Faithfulness Score (Chunk Size 256): {faithfulness2}")
logger.debug(f"Relevancy Score (Chunk Size 256): {relevancy2}")
faithfulness3, relevancy3 = evaluate_response(
    query, ai_responses_dict[128], true_answer)
logger.debug(f"\n")
logger.debug(f"Faithfulness Score (Chunk Size 128): {faithfulness3}")
logger.debug(f"Relevancy Score (Chunk Size 128): {relevancy3}")
save_file({
    "chunk_size_256": {"faithfulness": faithfulness2, "relevancy": relevancy2},
    "chunk_size_128": {"faithfulness": faithfulness3, "relevancy": relevancy3}
}, f"{generated_dir}/evaluation_scores.json")
logger.info(
    f"Saved evaluation scores to {generated_dir}/evaluation_scores.json")

logger.info("\n\n[DONE]", bright=True)
