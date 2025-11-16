import os
import json
import numpy as np
import pymupdf as fitz
from typing import List, Dict, Any, Callable, Optional, Tuple, TypedDict
from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.logger import CustomLogger
import re

DATA_DIR = f"{os.path.dirname(__file__)}/data"
# DOCS_PATH = f"{DATA_DIR}/search_results.json"
DOCS_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/data/AI_Information.pdf"
# LLM_MODEL = "llama-3.2-3b-instruct-4bit"
LLM_MODEL = None


class SearchResult(TypedDict):
    id: str
    rank: int | None
    doc_index: int
    score: float
    text: str
    metadata: Dict[str, Any]
    relevance_score: float | None


class ValidationData(TypedDict):
    question: str
    answer: str


class ChunkMetadata(TypedDict):
    doc_index: int
    header: str
    parent_header: str
    header_level: int


class Chunk(TypedDict):
    metadata: ChunkMetadata
    text: str


class ChunkData(TypedDict):
    chunks: List[Chunk]


class VectorStoreItem(TypedDict, total=False):
    id: str
    text: str
    metadata: Dict[str, Any]
    similarity: float


def setup_config(script_path: str) -> Tuple[str, str, str, CustomLogger]:
    script_dir = os.path.dirname(os.path.abspath(script_path))
    log_file = os.path.join(
        script_dir, f"{os.path.splitext(os.path.basename(script_path))[0]}.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    file_name = os.path.splitext(os.path.basename(script_path))[0]
    generated_dir = os.path.join(script_dir, "generated", file_name)
    os.makedirs(generated_dir, exist_ok=True)
    return script_dir, generated_dir, log_file, logger


def load_json_data(data_path: str, logger: CustomLogger) -> Tuple[List[str], List[Chunk]]:
    chunks = chunk_document(data_path)
    logger.debug(f"Number of text chunks: {len(chunks)}")
    logger.debug("\nFirst text chunk:")
    logger.debug(chunks[0])
    return [chunk["text"] for chunk in chunks], chunks


def initialize_llm(logger: CustomLogger) -> Tuple[LlamacppLLM, Callable[[str | List[str]], List[float] | List[List[float]]]]:
    logger.info("Initializing Llama.cpp LLM and embedding function")
    llm = LlamacppLLM(
        model="llama-3.2-3b-instruct-4bit",
        base_url="http://shawn-pc.local:8080/v1",
        verbose=False
    )
    embed_client = LlamacppEmbedding(
        model="embeddinggemma",
        base_url="http://shawn-pc.local:8081/v1",
        use_cache=True,
        cache_backend="sqlite"
    )
    embed_func = embed_client.get_embedding_function(return_format="numpy")
    return llm, embed_func


def generate_embeddings(
    texts: str | List[str],
    embed_func: Callable[[str | List[str]], List[float] | List[List[float]]],
    logger: CustomLogger
) -> np.ndarray | List[np.ndarray]:
    logger.info("Generating embeddings for chunks")
    embeddings = embed_func(texts)
    logger.info("Embeddings generated")
    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


def load_validation_data(val_path: str, logger: CustomLogger) -> List[ValidationData]:
    logger.info("Loading validation data")
    with open(val_path) as f:
        data: List[ValidationData] = json.load(f)
    return data


def generate_ai_response(
    query: str,
    system_prompt: str,
    retrieved_chunks: List[SearchResult],
    llm: LlamacppLLM,
    logger: CustomLogger,
    model: str = "llama-3.2-3b-instruct-4bit",
    **kwargs
) -> str:
    logger.info("Generating AI response")
    context = "\n".join([
        f"Context {i+1}:\n{chunk['text']}\n=====================================\n"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    user_prompt = f"{context}\nQuestion: {query}"
    response = ""
    for chunk in llm.chat_stream(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        **kwargs
    ):
        delta = chunk.choices[0].delta
        content = delta.content if delta.content is not None else ""
        response += content
        logger.success(content, flush=True)
    logger.success(f"AI Response:\n{response}")
    return response


def evaluate_ai_response(
    question: str,
    response: str,
    true_answer: str,
    llm: LlamacppLLM,
    logger: CustomLogger,
    model: str = "llama-3.2-3b-instruct-4bit",
    **kwargs
) -> Tuple[float, str]:
    logger.info("Evaluating response")
    evaluate_system_prompt = (
        "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. "
        "If the AI assistant's response is very close to the true response, assign a score of 1. "
        "If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. "
        "If the response is partially aligned with the true response, assign a score of 0.5."
    )

    def parse_score(text: str) -> float:
        match = re.search(r'-?\d*\.?\d+', text)
        if match:
            return float(match.group())
        raise ValueError(f"No valid float found in text: {text}")
    evaluation_prompt = f"User Query: {question}\nAI Response:\n{response}\nTrue Response: {true_answer}\n{evaluate_system_prompt}"
    response = ""
    for chunk in llm.chat_stream(
        messages=[
            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
            {"role": "user", "content": evaluation_prompt}
        ],
        **kwargs
    ):
        delta = chunk.choices[0].delta
        content = delta.content if delta.content is not None else ""
        response += content
        logger.success(content, flush=True)
    try:
        score = parse_score(response.strip())
    except ValueError:
        logger.debug(
            "Warning: Could not parse evaluation score, defaulting to 0")
        score = 0.0
    return score, response


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and prints the first `num_chars` characters.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
        all_text += text  # Append the extracted text to the all_text string

    return all_text  # Return the extracted text


def chunk_text(text, n, overlap, doc_index=0):
    """
    Chunks the given text into segments of n characters with overlap.

    Args:
    text (str): The text to be chunked.
    n (int): The number of characters in each chunk.
    overlap (int): The number of overlapping characters between chunks.
    doc_index (int): The document index to include in metadata.

    Returns:
    List[dict]: A list of dicts with 'text' and 'metadata' (containing 'doc_index', 'chunk_id', 'start_char', 'end_char').
    """
    chunks = []
    chunk_id = 1
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        if chunk:
            start_char = i
            end_char = i + len(chunk)
            chunks.append({
                "text": chunk,
                "metadata": {
                    "doc_index": doc_index,
                    "chunk_id": chunk_id,
                    "start_char": start_char,
                    "end_char": end_char
                }
            })
            chunk_id += 1
    return chunks


def chunk_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for use with adaptive retrieval.

    Args:
    pdf_path (str): Path to the PDF file.
    chunk_size (int): Size of each chunk in characters.
    chunk_overlap (int): Overlap between chunks in characters.

    Returns:
    Tuple[List[str], SimpleVectorStore]: Document chunks and vector store.
    """
    # Extract text from the PDF file
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # Chunk the extracted text
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")

    # Return the chunks
    return chunks


class SimpleVectorStore:
    def __init__(self) -> None:
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []

    def add_item(self, text: str, embedding: List[float] | np.ndarray, metadata: Dict[str, Any] | None = None, id: str | None = None) -> None:
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
        self.ids.append(id if id is not None else str(len(self.ids)))

    def search(self, query_embedding: List[float] | np.ndarray, top_k: int = 5, filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[VectorStoreItem]:
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding).flatten()
        similarities: List[Tuple[int, float]] = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
            dot_product = np.dot(query_vector, vector)
            query_norm = np.linalg.norm(query_vector)
            vector_norm = np.linalg.norm(vector)
            similarity = 0.0 if query_norm == 0 or vector_norm == 0 else dot_product / \
                (query_norm * vector_norm)
            if filter_func is None or filter_func(self.metadata[i]):
                similarities.append((i, similarity))
        similarities.sort(key=lambda x: -float('inf')
                          if np.isnan(x[1]) else x[1], reverse=True)
        results: List[VectorStoreItem] = []
        for i in range(min(top_k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "id": self.ids[idx],
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        return results
