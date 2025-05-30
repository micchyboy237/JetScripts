import re
from jet.llm.mlx.mlx_types import EmbedModelType
from jet.llm.utils.transformer_embeddings import generate_embeddings
from transformers import AutoTokenizer
import numpy as np
from typing import List, Optional, TypedDict
from jet.llm.mlx.models import AVAILABLE_EMBED_MODELS, resolve_model_key
from jet.logger import logger


class SimilarityResult(TypedDict):
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int


def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing, removing non-alphanumeric characters, and normalizing whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def search_docs(
    query: str,
    documents: List[str],
    model: EmbedModelType = "all-minilm:33m",
    top_k: Optional[int] = 10,
    batch_size: Optional[int] = None,
    normalize: bool = True,
    chunk_size: Optional[int] = None,
    ids: Optional[List[str]] = None,
    preprocess: bool = True
) -> List[SimilarityResult]:
    """Search documents with memory-efficient embedding generation and return SimilarityResult."""
    if not query or not documents:
        raise ValueError("Query string and documents list must not be empty.")

    if not top_k:
        top_k = len(documents)

    if ids is not None:
        if len(ids) != len(documents):
            raise ValueError(
                f"Length of ids ({len(ids)}) must match length of documents ({len(documents)})")
        if len(ids) != len(set(ids)):
            raise ValueError("IDs must be unique")

    if preprocess:
        processed_query = preprocess_text(query)
        processed_documents = [preprocess_text(doc) for doc in documents]
    else:
        processed_query = query
        processed_documents = documents

    embed_model = resolve_model_key(model)
    model_id = AVAILABLE_EMBED_MODELS[embed_model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    query_embedding = generate_embeddings(
        model, processed_query, batch_size, normalize, chunk_size=chunk_size)
    doc_embeddings = generate_embeddings(
        model, processed_documents, batch_size, normalize, chunk_size=chunk_size)

    query_embedding = np.array(query_embedding)
    doc_embeddings = np.array(doc_embeddings)

    if len(doc_embeddings) == 0 or len(processed_documents) == 0:
        return []
    if len(doc_embeddings) != len(processed_documents):
        logger.error(
            f"Mismatch between document embeddings ({len(doc_embeddings)}) and documents ({len(processed_documents)})")
        return []

    # Compute similarities with zero-norm handling
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    similarities = np.dot(doc_embeddings, query_embedding)
    # Avoid division by zero
    valid_norms = (doc_norms > 0) & (query_norm > 0)
    similarities[valid_norms] /= (doc_norms[valid_norms] * query_norm)
    similarities[~valid_norms] = -1.0  # Replace NaN/invalid with -1.0

    top_k = min(top_k, len(processed_documents))
    if top_k <= 0:
        return []

    top_indices = np.argsort(similarities)[::-1][:top_k]
    valid_indices = [int(idx)
                     for idx in top_indices if idx < len(processed_documents)]
    if not valid_indices:
        return []

    results = []
    for rank, idx in enumerate(valid_indices, start=1):
        doc_text = documents[idx]
        tokens = len(tokenizer.encode(doc_text, add_special_tokens=True))
        doc_id = ids[idx] if ids is not None else f"doc_{idx}"
        result = SimilarityResult(
            id=doc_id,
            rank=rank,
            doc_index=int(idx),
            score=float(similarities[idx]),
            text=doc_text,
            tokens=tokens
        )
        results.append(result)

    return results


def main():
    """Example usage of search_docs function."""
    # Sample query and documents
    query = "Machine learning algorithms"
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
        "Deep learning is a type of machine learning that uses neural networks with many layers.",
        "Python is a popular programming language for machine learning and data science.",
        "The history of artificial intelligence began in the 1950s with early computational models."
    ]
    ids = ["doc1", "doc2", "doc3", "doc4"]

    try:
        # Basic search with default parameters
        print("Performing basic search...")
        results = search_docs(query, documents, ids=ids)
        for result in results:
            print(
                f"ID: {result['id']}, Rank: {result['rank']}, Score: {result['score']:.4f}")
            print(f"Text: {result['text']}")
            print(f"Tokens: {result['tokens']}")
            print("---")

        # Search with specific parameters
        print("\nPerforming search with top_k=2 and preprocess=False...")
        results = search_docs(
            query,
            documents,
            model="all-minilm:33m",
            top_k=2,
            preprocess=False,
            ids=ids
        )
        for result in results:
            print(
                f"ID: {result['id']}, Rank: {result['rank']}, Score: {result['score']:.4f}")
            print(f"Text: {result['text']}")
            print(f"Tokens: {result['tokens']}")
            print("---")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
