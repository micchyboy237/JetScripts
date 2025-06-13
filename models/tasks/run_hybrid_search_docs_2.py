from typing import Union, List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Callable, Union, TypedDict
from jet.models.embeddings.base import generate_embeddings
from jet.vectors.document_types import HeaderDocument
from jet.file.utils import load_file, save_file
import os
from unittest.mock import Mock
import pytest
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from jet.vectors.document_types import HeaderDocument, HeaderMetadata
from jet.logger import logger
from jet.wordnet.text_chunker import chunk_headers


def search_docs(
    query: str,
    docs: List[HeaderDocument],
    embed_func: Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]],
    top_k: Optional[int] = 5
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on HeaderDocument objects using text and header embeddings.

    Args:
        query: Search query string.
        docs: List of HeaderDocument objects to search.
        embed_func: Function to generate embeddings for text, accepting a string or list of strings
                   and returning a list of floats or list of list of floats.
        top_k: Number of top results to return (None for all).

    Returns:
        List of dictionaries containing search results with scores and metadata.
    """
    logger.debug("Starting search_docs with query: %s, %d docs",
                 query, len(docs))

    if not docs:
        logger.info("No documents provided, returning empty results")
        return []

    # Collect all texts to embed in one batch
    texts_to_embed = [query]  # Start with query
    doc_texts = []
    header_texts = []
    for i, doc in enumerate(docs):
        metadata = HeaderMetadata(**doc.metadata)
        doc_texts.append(doc.text)
        header = metadata.get("header", "")
        # Fallback to text if no header
        header_texts.append(header if header else doc.text)
        texts_to_embed.extend([doc.text, header_texts[-1]])

    logger.debug("Collected %d texts for batch embedding", len(texts_to_embed))

    # Generate all embeddings in one batch
    try:
        all_embeddings = embed_func(texts_to_embed)
        if not isinstance(all_embeddings, list) or not all_embeddings:
            logger.error("Invalid embeddings returned from embed_func")
            raise ValueError(
                "embed_func must return a non-empty list of embeddings")

        # Split embeddings: query (1), doc texts (N), headers (N)
        query_embedding = np.array(all_embeddings[0])
        doc_embeddings = np.array(all_embeddings[1::2])  # Odd indices
        header_embeddings = np.array(all_embeddings[2::2])  # Even indices
        logger.debug("Embeddings shapes: query=%s, docs=%s, headers=%s",
                     query_embedding.shape, doc_embeddings.shape, header_embeddings.shape)

    except Exception as e:
        logger.error("Failed to generate embeddings: %s", str(e))
        raise

    # Calculate cosine similarities
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        logger.error("Query embedding has zero norm")
        raise ValueError("Query embedding is invalid (zero norm)")

    results = []
    for i, (doc, text_embedding, header_embedding) in enumerate(zip(docs, doc_embeddings, header_embeddings)):
        metadata = HeaderMetadata(**doc.metadata)

        # Normalize embeddings for cosine similarity
        text_norm = np.linalg.norm(text_embedding)
        header_norm = np.linalg.norm(header_embedding)

        # Calculate similarities
        sim_text = np.dot(query_embedding, text_embedding) / \
            (query_norm * text_norm) if text_norm > 0 else 0.0
        sim_header = np.dot(query_embedding, header_embedding) / \
            (query_norm * header_norm) if header_norm > 0 else sim_text
        avg_score = (sim_text + sim_header) / 2

        result = {
            "id": doc.id,
            "rank": None,  # Will be set after sorting
            "doc_index": metadata.get("doc_index", 0),
            "chunk_index": metadata.get("chunk_index", 0),
            "source_url": metadata.get("source_url", None),
            "tokens": metadata.get("tokens", None),
            "score": float(avg_score),
            "text": doc.text,
            "header": header_texts[i],
            "parent_header": metadata.get("parent_header", ""),
            "header_level": metadata.get("header_level", 0),
            "embedding": text_embedding.tolist(),
            "header_embedding": header_embedding.tolist()
        }
        results.append((result, avg_score))
        logger.debug("Processed doc %s: score=%.4f", doc.id, avg_score)

    # Sort and assign ranks
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:top_k] if top_k else results
    for rank, (result, _) in enumerate(top_results, 1):
        result["rank"] = rank

    logger.info("Found %d top results for query", len(top_results))
    return [result for result, _ in top_results]


# Pytest tests

class TestSearchDocs:
    def test_search_docs(self):
        embed_func = Mock(side_effect=lambda x: [
                          1.0, 0.0] if "query" in x else [0.0, 1.0])
        doc = HeaderDocument(
            id="doc1",
            text="Test content",
            metadata={"header": "Test header", "doc_index": 1}
        )
        result = search_docs("query", [doc], embed_func, top_k=1)
        expected = [{
            "id": "doc1",
            "rank": 1,
            "doc_index": 1,
            "chunk_index": 0,
            "source_url": None,
            "score": 0.0,  # Cosine similarity of orthogonal vectors
            "text": "Test content",
            "header": "Test header",
            "parent_header": "",
            "header_level": 0,
            "embedding": [0.0, 1.0],
            "header_embedding": [0.0, 1.0]
        }]
        assert result == expected
        logger.debug("Test search_docs passed")


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Output directory: %s", output_dir)

    query = "List all ongoing and upcoming isekai anime 2025."
    logger.debug("Loading documents from %s", docs_file)
    docs = load_file(docs_file)
    docs = [HeaderDocument(**doc) for doc in docs]
    logger.info("Loaded %d documents", len(docs))

    # Chunk documents using chunk_headers
    chunked_docs = chunk_headers(docs, max_tokens=200)
    logger.info("Chunked into %d documents", len(chunked_docs))

    # Save chunked documents
    save_file([{
        "id": doc.id,
        "text": doc.text
    } for doc in chunked_docs], f"{output_dir}/chunked_docs.json")
    logger.debug("Saved chunked documents to %s/chunked_docs.json", output_dir)

    # Perform semantic search
    model_name = "static-retrieval-mrl-en-v1"

    def embed_func(x): return generate_embeddings(
        x, model_name, show_progress=True)
    search_results = search_docs(
        query, chunked_docs, embed_func, top_k=10)

    for result in search_results:
        # Extract the base ID from the chunked document's ID
        base_id = result['id'].split('_chunk_')[0]
        logger.debug(
            "Searching for original document with base ID: %s", base_id)
        logger.debug("Available doc IDs: %s", [doc.id for doc in docs])
        try:
            original_doc = next(doc for doc in docs if doc.id == base_id)
            logger.success(
                f"\nRank {result['rank']} (Doc: {result['doc_index']} | Chunk: {result['chunk_index']} | Tokens: {result['tokens']}):")
            print(f"Score: {result['score']:.4f}")
            print(f"Header: {result['header']}")
            print(f"Parent Header: {result['parent_header']}")
            print(f"Chunk:\n{result['text']}")
            print(f"Original Document:\n{original_doc.text}")
        except StopIteration:
            logger.error(f"No original document found for base ID: {base_id}")
            continue

    save_file(search_results, f"{output_dir}/results.json")
    logger.info("Saved search results to %s/results.json", output_dir)
