from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
from tqdm import tqdm
import pypdf
import json
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

DATA_DIR = os.path.join(script_dir, "data")

logger.info("Initializing PDF extraction")


def extract_text_from_pdf(pdf_path):
    all_text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_text += text
    return all_text


pdf_path = f"{DATA_DIR}/AI_Information.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
logger.debug(extracted_text[:500])

logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")

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

logger.info("Generating embeddings for chunks")


def create_embeddings(texts):
    return embed_func(texts)


chunk_embeddings_dict = {size: create_embeddings(chunks) for size, chunks in tqdm(
    text_chunks_dict.items(), desc="Generating Embeddings")}

logger.info("Defining similarity and retrieval functions")


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve_relevant_chunks(query, text_chunks, chunk_embeddings, k=5):
    query_embedding = embed_func(query)
    similarities = [cosine_similarity(query_embedding, emb)
                    for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [text_chunks[i] for i in top_indices]


with open(f"{DATA_DIR}/val.json") as f:
    data = json.load(f)
query = data[3]['question']
retrieved_chunks_dict = {size: retrieve_relevant_chunks(
    query, text_chunks_dict[size], chunk_embeddings_dict[size]) for size in chunk_sizes}
logger.debug(retrieved_chunks_dict[256])

logger.info("Generating AI responses")
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"


def generate_response(query, system_prompt, retrieved_chunks, model="meta-llama/Llama-3.2-3B-Instruct"):
    context = "\n".join(
        [f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
    user_prompt = f"{context}\n\nQuestion: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    return response


ai_responses_dict = {size: generate_response(
    query, system_prompt, retrieved_chunks_dict[size]) for size in chunk_sizes}
logger.debug(ai_responses_dict[256])

logger.info("Evaluating responses")

SCORE_FULL = 1.0     # Complete match or fully satisfactory
SCORE_PARTIAL = 0.5  # Partial match or somewhat satisfactory
SCORE_NONE = 0.0     # No match or unsatisfactory

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
        faithfulness_score = float(faithfulness_response.strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse faithfulness score, defaulting to 0")
        faithfulness_score = 0.0
    try:
        relevancy_score = float(relevancy_response.strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse relevancy score, defaulting to 0")
        relevancy_score = 0.0
    return faithfulness_score, relevancy_score


true_answer = data[3]['ideal_answer']

# faithfulness, relevancy = evaluate_response(
#     query, ai_responses_dict[512], true_answer)
# logger.debug(f"\n")
# logger.debug(f"Faithfulness Score (Chunk Size 512): {faithfulness}")
# logger.debug(f"Relevancy Score (Chunk Size 512): {relevancy}")

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

logger.info("\n\n[DONE]", bright=True)
