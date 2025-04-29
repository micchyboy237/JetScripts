from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
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
GENERATED_DIR = os.path.join("results", file_name)
DATA_DIR = os.path.join(script_dir, "data")
os.makedirs(GENERATED_DIR, exist_ok=True)

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

logger.info("Splitting text into sentences")
sentences = extracted_text.split(". ")
logger.debug(f"Number of sentences: {len(sentences)}")

logger.info("Generating sentence embeddings")


def get_embedding(text):
    return embed_func(text)


embeddings = embed_func(sentences)
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

logger.info("Creating embeddings for chunks")


chunk_embeddings = embed_func(text_chunks)

logger.info("Performing semantic search")


def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    query_embedding = get_embedding(query)
    similarities = [cosine_similarity(query_embedding, emb)
                    for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [text_chunks[i] for i in top_indices]


with open(f"{DATA_DIR}/val.json") as f:
    data = json.load(f)
query = data[0]['question']
top_chunks = semantic_search(query, text_chunks, chunk_embeddings, k=2)
logger.debug(f"Query: {query}")
for i, chunk in enumerate(top_chunks):
    logger.debug(f"Context {i+1}:\n{chunk}\n{'='*40}")

logger.info("Generating AI response")
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"


def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )
    return response


user_prompt = "\n".join(
    [f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"
ai_response = generate_response(system_prompt, user_prompt)
logger.success(ai_response)

logger.info("Evaluating response")
evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"
evaluation_response = generate_response(
    evaluate_system_prompt, evaluation_prompt)
logger.success(evaluation_response)

logger.info("\n\n[DONE]", bright=True)
