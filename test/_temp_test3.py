import json
import os
import re
from typing import List, Tuple, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import CompletionResponse
from numpy.typing import NDArray
import numpy as np
from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
from tqdm import tqdm
from typing import Final

# TypedDict definitions for JSON data structure


class MetadataDict(TypedDict):
    doc_index: int
    header_level: int
    header: str
    parent_header: Optional[str]


class HeaderDict(TypedDict):
    tokens: int
    text: str
    metadata: MetadataDict


class JsonDataDict(TypedDict):
    file: str
    header_count: int
    min_tokens: int
    max_tokens: int
    headers: List[HeaderDict]


# Customizable Variables
# File Paths
SCRIPT_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR: Final[str] = os.path.join(SCRIPT_DIR, "data")
DATA_DIR: Final[str] = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/data/hybrid_reranker_data/anime/top_isekai_anime"
GENERATED_DIR_NAME: Final[str] = "results"
JSON_PATH: Final[str] = os.path.join(DATA_DIR, "web_scraped_data.json")
VAL_JSON_PATH: Final[str] = os.path.join(DATA_DIR, "val.json")
LOG_FILE_NAME: Final[str] = f"{os.path.splitext(os.path.basename(__file__))[0]}.log"
LOG_FILE: Final[str] = os.path.join(SCRIPT_DIR, LOG_FILE_NAME)
MLX_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), os.path.splitext(os.path.basename(__file__))[0])

# Chunking Parameters
CHUNK_SIZE: Final[int] = 1000  # Max characters per chunk
CHUNK_OVERLAP: Final[int] = 200  # Overlap between chunks

# Embedding Parameters
BATCH_SIZE: Final[int] = 32  # Batch size for embedding generation
EMBEDDING_MODEL: Final[str] = "mxbai-embed-large"  # Embedding model

# Search Parameters
K: Final[int] = 1  # Number of top chunks to retrieve
CONTEXT_SIZE: Final[int] = 1  # Number of neighboring chunks to include
HEADER_BOOST: Final[float] = 1.2  # Score multiplier for header match
# Score multiplier for parent_header match
PARENT_HEADER_BOOST: Final[float] = 1.1

# Model Parameters
# Model for response generation
RESPONSE_MODEL: Final[str] = "meta-llama/Llama-3.2-3B-Instruct"

# Query
QUERY: Final[str] = "What is No Game No Life about?"  # Example query

# Setup logging
logger: CustomLogger = CustomLogger(LOG_FILE, overwrite=True)
logger.info(f"Logs: {LOG_FILE}")

# Create generated directory
GENERATED_DIR: str = os.path.join(
    GENERATED_DIR_NAME, os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(GENERATED_DIR, exist_ok=True)

logger.info("Initializing JSON data extraction")

# Load and clean JSON data


def load_json_data(json_path: str) -> Tuple[List[HeaderDict], List[str]]:
    with open(json_path, 'r') as f:
        data: JsonDataDict = json.load(f)
    headers: List[HeaderDict] = data.get('headers', [])
    all_text: List[str] = []
    for item in headers:
        text: str = clean_text(item.get('text', ''))
        all_text.append(text)
    return headers, all_text


def clean_text(text: str) -> str:
    # Remove Markdown headers (#, ##), navigation symbols (>, *), and normalize whitespace
    # Remove Markdown headers
    text = re.sub(r'^#{1,}\s*', '', text, flags=re.MULTILINE)
    # Replace >, *, and newlines with space
    text = re.sub(r'[\>\*\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


json_data: List[HeaderDict]
extracted_text: List[str]
json_data, extracted_text = load_json_data(JSON_PATH)
logger.debug(f"Loaded {len(json_data)} JSON entries")
logger.debug(extracted_text[0][:500])

logger.info("Initializing MLX and embedding function")
mlx: MLX = MLX(log_dir=MLX_LOG_DIR)
embed_func = get_embedding_function(EMBEDDING_MODEL)

logger.info("Using JSON entries as chunks")
# Use JSON entries as chunks directly
text_chunks: List[str] = extracted_text
logger.debug(f"Number of text chunks: {len(text_chunks)}")
logger.debug("\nFirst text chunk:")
logger.debug(text_chunks[0])

# Optional: Chunk large entries if needed


def chunk_text(text: str, n: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks

# Example: Uncomment to chunk entries > CHUNK_SIZE
# text_chunks: List[str] = []
# for text in extracted_text:
#     if len(text) > CHUNK_SIZE:
#         text_chunks.extend(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP))
#     else:
#         text_chunks.append(text)


logger.info("Generating embeddings for chunks")


def create_embeddings(texts: List[str], batch_size: int) -> List[NDArray[np.float64]]:
    embeddings: List[NDArray[np.float64]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch: List[str] = texts[i:i + batch_size]
        embeddings.extend(embed_func(batch))
    return embeddings


response: List[NDArray[np.float64]] = create_embeddings(
    text_chunks, BATCH_SIZE)
logger.info("Embeddings generated")

logger.info("Defining similarity and context-enriched search functions")


def cosine_similarity(vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> float:
    norm1: float = np.linalg.norm(vec1)
    norm2: float = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Handle zero vector case
    return np.dot(vec1, vec2) / (norm1 * norm2)


def context_enriched_search(
    query: str,
    json_data: List[HeaderDict],
    embeddings: List[NDArray[np.float64]],
    k: int,
    context_size: int
) -> List[str]:
    query_embedding: NDArray[np.float64] = embed_func(query)
    similarity_scores: List[Tuple[int, float]] = []
    for i, (chunk_embedding, item) in enumerate(zip(embeddings, json_data)):
        similarity_score: float = cosine_similarity(
            np.array(query_embedding), np.array(chunk_embedding))
        # Boost score if header or parent_header matches query
        metadata: MetadataDict = item.get('metadata', {})
        header: str = metadata.get('header', '').lower()
        parent_header: Optional[str] = metadata.get('parent_header')
        query_lower: str = query.lower()
        if header and header in query_lower:
            similarity_score *= HEADER_BOOST  # Boost for header match
        elif parent_header and parent_header.lower() in query_lower:
            similarity_score *= PARENT_HEADER_BOOST  # Boost for parent_header match
        similarity_scores.append((i, similarity_score))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_index: int = similarity_scores[0][0]
    start: int = max(0, top_index - context_size)
    end: int = min(len(json_data), top_index + context_size + 1)
    return [json_data[i].get('text', '').strip() for i in range(start, end)]


top_chunks: List[str] = context_enriched_search(
    QUERY, json_data, response, k=K, context_size=CONTEXT_SIZE)
logger.debug(f"Query: {QUERY}")
for i, chunk in enumerate(top_chunks):
    logger.debug(
        f"Context {i + 1}:\n{chunk}\n=====================================")

logger.info("Generating AI response")
system_prompt: Final[str] = (
    "You are an AI assistant that strictly answers based on the given context. "
    "If the answer cannot be derived directly from the provided context, respond with: "
    "'I do not have enough information to answer that.'"
)


def generate_response(
    query: str,
    system_prompt: str,
    retrieved_chunks: List[str],
    model: str
) -> str:
    context: str = "\n".join(
        [f"Context {i+1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(retrieved_chunks)])
    user_prompt: str = f"{context}\nQuestion: {query}"
    response: str = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model
    )
    return response


ai_response: str = generate_response(
    QUERY, system_prompt, top_chunks, RESPONSE_MODEL)
logger.debug(f"AI Response: {ai_response}")

logger.info("Evaluating response")
evaluate_system_prompt: Final[str] = (
    "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. "
    "If the AI assistant's response is very close to the true response, assign a score of 1. "
    "If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. "
    "If the response is partially aligned with the true response, assign a score of 0.5."
)


class ValEntryDict(TypedDict):
    question: str
    answer: str


def evaluate_response(question: str, response: str, true_answer: str) -> float:
    evaluation_prompt: str = (
        f"User Query: {question}\nAI Response:\n{response}\n"
        f"True Response: {true_answer}\n{evaluate_system_prompt}"
    )
    evaluation_response: CompletionResponse = mlx.chat(
        [
            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
            {"role": "user", "content": evaluation_prompt}
        ]
    )
    try:
        score: float = float(str(evaluation_response).strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse evaluation score, defaulting to 0")
        score = 0.0
    return score


logger.info("Loading validation data for evaluation")
with open(VAL_JSON_PATH) as f:
    val_data: List[ValEntryDict] = json.load(f)
# Assumes answer for first entry
true_answer: str = val_data[0]['answer']
evaluation_score: float = evaluate_response(QUERY, ai_response, true_answer)
logger.debug(f"Evaluation Score: {evaluation_score}")

logger.info("\n\n[DONE]", bright=True)
