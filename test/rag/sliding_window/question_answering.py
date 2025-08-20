from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from utils import sliding_window_split, TextChunk
import mlx.core as mx
from mlx_lm import load, generate


def embed_chunks(chunks: List[TextChunk], embedder: SentenceTransformer) -> List[np.ndarray]:
    """Embed text chunks using a sentence transformer model."""
    return [embedder.encode(chunk["text"]) for chunk in chunks]


def retrieve_relevant_chunk(query: str, chunks: List[TextChunk], embedder: SentenceTransformer) -> TextChunk:
    """Retrieve the most relevant chunk for a query using cosine similarity."""
    query_embedding = embedder.encode(query)
    chunk_embeddings = embed_chunks(chunks, embedder)
    similarities = [np.dot(query_embedding, emb) / (np.linalg.norm(
        query_embedding) * np.linalg.norm(emb)) for emb in chunk_embeddings]
    return chunks[np.argmax(similarities)]


def answer_question(text: str, query: str, model_path: str = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125") -> str:
    """Answer a question over a large text using sliding windows and RAG."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = sliding_window_split(text)
    relevant_chunk = retrieve_relevant_chunk(query, chunks, embedder)
    model, tokenizer = load(model_path)
    prompt = f"Based on the following text, answer the question: {query}\nText: {relevant_chunk['text']}"
    answer = generate(model, tokenizer, prompt=prompt, max_tokens=150)
    return answer.strip()


if __name__ == "__main__":
    sample_text = "AI is advancing rapidly. Machine learning improves predictions. Deep learning excels in image recognition." * 100
    query = "What does deep learning excel in?"
    answer = answer_question(sample_text, query)
    print("Answer:", answer)
