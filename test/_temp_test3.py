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

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("generated", file_name)
# DATA_DIR = os.path.join(script_dir, "data")
DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/data/hybrid_reranker_data/anime/top_isekai_anime"
os.makedirs(GENERATED_DIR, exist_ok=True)

logger.info("Initializing PDF extraction")


def extract_text_from_pdf(pdf_path):
    # all_text = ""
    # with open(pdf_path, "rb") as file:
    #     reader = pypdf.PdfReader(file)
    #     for page in reader.pages:
    #         text = page.extract_text() or ""
    #         all_text += text
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_isekai_anime.md"
    all_text = load_file(data_path)
    return all_text


pdf_path = f"{DATA_DIR}/AI_Information.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
logger.debug(extracted_text[:500])

logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")

logger.info("Chunking text")


def chunk_text(text, n, overlap):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks


text_chunks = chunk_text(extracted_text, 1000, 200)
logger.debug(f"Number of text chunks: {len(text_chunks)}")
logger.debug("\nFirst text chunk:")
logger.debug(text_chunks[0])

logger.info("Generating embeddings for chunks")


def create_embeddings(texts):
    return embed_func(texts)


response = create_embeddings(text_chunks)
logger.info("Embeddings generated")

logger.info("Defining similarity and context-enriched search functions")


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    query_embedding = embed_func(query)
    similarity_scores = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(
            np.array(query_embedding), np.array(chunk_embedding))
        similarity_scores.append((i, similarity_score))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_index = similarity_scores[0][0]
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size + 1)
    return [text_chunks[i] for i in range(start, end)]


logger.info("Loading validation data")
with open(f"{DATA_DIR}/val.json") as f:
    data = json.load(f)

query = data[0]['question']
top_chunks = context_enriched_search(
    query, text_chunks, response, k=1, context_size=1)
logger.debug(f"Query: {query}")
for i, chunk in enumerate(top_chunks):
    logger.debug(
        f"Context {i + 1}:\n{chunk}\n=====================================")
save_file(top_chunks, f"{GENERATED_DIR}/top_chunks.json")

logger.info("Generating AI response")
system_prompt = "You are a helpful AI Assistance that can read structured and unstructured texts with headers (lines that start with #). You strictly answer based on the given context."


def generate_response(query, system_prompt, retrieved_chunks, model="meta-llama/Llama-3.2-3B-Instruct"):
    context = "\n".join(
        [f"Context {i+1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(retrieved_chunks)])
    user_prompt = f"{context}\nQuestion: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response["content"]


ai_response = generate_response(query, system_prompt, top_chunks)
logger.debug(f"AI Response: {ai_response}")

logger.info("Evaluating response")
evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."


def evaluate_response(question, response, true_answer):
    def parse_score(text: str) -> float:
        # Find the first occurrence of a float in the text (e.g., 0.5, -1.23, 42.0)
        match = re.search(r'-?\d*\.?\d+', text)
        if match:
            return float(match.group())
        raise ValueError(f"No valid float found in text: {text}")

    evaluation_prompt = f"User Query: {question}\nAI Response:\n{response}\nTrue Response: {true_answer}\n{evaluate_system_prompt}"
    evaluation_response = mlx.chat(
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
    return score


true_answer = data[0]['answer']
evaluation_score = evaluate_response(query, ai_response, true_answer)
logger.debug(f"Evaluation Score: {evaluation_score}")

logger.info("\n\n[DONE]", bright=True)
