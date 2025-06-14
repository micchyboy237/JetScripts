from typing import Union, List, Callable, Optional, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.vectors.document_types import HeaderDocument, HeaderMetadata
from jet.file.utils import load_file, save_file
from jet.wordnet.text_chunker import chunk_headers
import os
from unittest.mock import Mock
import pytest
import logging
import numpy as np
import re


def search_docs(
    query: str,
    docs: List[HeaderDocument],
    embed_func: Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]],
    top_k: Optional[int] = 5,
    header_weight: float = 0.5,
    content_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on HeaderDocument objects using text and header embeddings.

    Args:
        query: Search query string.
        docs: List of HeaderDocument objects to search.
        embed_func: Function to generate embeddings for text, accepting a string or list of strings
                   and returning a list of floats or list of list of floats.
        top_k: Number of top results to return (None for all).
        header_weight: Weight for header similarity in final score (default: 0.5).
        content_weight: Weight for content similarity in final score (default: 0.5).

    Returns:
        List of dictionaries containing search results with scores and metadata.
    """
    logger.debug("Starting search_docs with query: %s, %d docs, weights: header=%.2f, content=%.2f",
                 query, len(docs), header_weight, content_weight)

    if not docs:
        logger.info("No documents provided, returning empty results")
        return []

    if not (0 <= header_weight <= 1 and 0 <= content_weight <= 1):
        logger.error("Weights must be between 0 and 1")
        raise ValueError("Weights must be between 0 and 1")

    if abs(header_weight + content_weight - 1.0) > 1e-6:
        logger.error("Weights must sum to 1")
        raise ValueError("Weights must sum to 1")

    # Collect all texts to embed in one batch
    texts_to_embed = [query]  # Start with query
    doc_texts = []
    header_texts = []
    for i, doc in enumerate(docs):
        metadata = HeaderMetadata(**doc.metadata)
        # Fallback to doc.text if no content
        content = metadata.get("content", doc.text)
        doc_texts.append(content)
        header = metadata.get("header", "")  # Empty string if no header
        header_texts.append(header)
        texts_to_embed.extend([content, header])

    logger.debug("Collected %d texts for batch embedding", len(texts_to_embed))

    # Generate all embeddings in one batch
    try:
        all_embeddings = embed_func(texts_to_embed)
        if not isinstance(all_embeddings, list) or not all_embeddings:
            logger.error("Invalid embeddings returned from embed_func")
            raise ValueError(
                "embed_func must return a non-empty list of embeddings")

        query_embedding = np.array(all_embeddings[0])
        # Odd indices (content)
        doc_embeddings = np.array(all_embeddings[1::2])
        header_embeddings = np.array(
            all_embeddings[2::2])  # Even indices (header)
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

        # Combine scores with weights
        final_score = (
            header_weight * sim_header +
            content_weight * sim_text
        )

        result = {
            "id": doc.id,
            "rank": None,  # Will be set after sorting
            "doc_index": metadata.get("doc_index", 0),
            "chunk_index": metadata.get("chunk_index", 0),
            "source_url": metadata.get("source_url", None),
            "tokens": metadata.get("tokens", None),
            "score": float(final_score),
            "text": doc.text,
            "header": header_texts[i],
            "parent_header": metadata.get("parent_header", ""),
            "header_level": metadata.get("header_level", 0),
            "sim_text": float(sim_text),
            "sim_header": float(sim_header),
        }
        results.append((result, final_score))
        logger.debug("Processed doc %s: score=%.4f (text=%.4f, header=%.4f)",
                     doc.id, final_score, sim_text, sim_header)

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
        # Mock embed_func to return orthogonal vectors for query and header/content
        embed_func = Mock(side_effect=lambda x:
                          [1.0, 0.0] if "query" in x else
                          [0.0, 1.0]  # Header and content are the same
                          )
        doc = HeaderDocument(
            id="doc1",
            text="Test content",
            metadata={"header": "Test header", "doc_index": 1}
        )
        result = search_docs(
            "query", [doc], embed_func, top_k=1,
            header_weight=0.5, content_weight=0.5,
        )
        expected = [{
            "id": "doc1",
            "rank": 1,
            "doc_index": 1,
            "chunk_index": 0,
            "source_url": None,
            "tokens": None,
            # 0.5 * 0 (query-content) + 0.5 * 0 (query-header)
            "score": 0.0,
            "text": "Test content",
            "header": "Test header",
            "parent_header": "",
            "header_level": 0,
            "sim_text": 0.0,
            "sim_header": 0.0,
        }]
        assert result == expected
        logger.debug("Test search_docs passed")

    def test_search_docs_invalid_weights(self):
        embed_func = Mock(side_effect=lambda x: [
            1.0, 0.0] if "query" in x else [0.0, 1.0])
        doc = HeaderDocument(
            id="doc1",
            text="Test content",
            metadata={"header": "Test header", "doc_index": 1}
        )
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            search_docs(
                "query", [doc], embed_func, top_k=1,
                header_weight=0.7, content_weight=0.5
            )
        logger.debug("Test search_docs_invalid_weights passed")


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Output directory created: %s", output_dir)

    query = "List all ongoing and upcoming isekai anime 2025."
    logger.debug("Loading documents from %s", docs_file)
    docs = load_file(docs_file)
    docs = [HeaderDocument(**doc) for doc in docs]
    logger.info("Loaded %d documents", len(docs))

    chunked_docs = chunk_headers(docs, max_tokens=200)
    logger.info("Chunked into %d documents", len(chunked_docs))

    # Fix headers for chunks
    header_pattern = re.compile(r'^#+ .+$', re.MULTILINE)
    doc_to_header = {}
    fixed_chunked_docs = []

    for doc in chunked_docs:
        base_id = doc.id.split('_chunk_')[0]
        chunk_index = doc.metadata.get("chunk_index", 0)
        logger.debug("Processing chunk ID: %s, Chunk Index: %d, Header: %s, Text: %s",
                     doc.id, chunk_index, doc.metadata.get("header", ""), doc.text[:50])

        # If first chunk, check for header in text
        if chunk_index == 0:
            match = header_pattern.match(doc.text.strip())
            if match:
                doc_to_header[base_id] = match.group(0).strip()
                logger.debug("Matched header for doc %s: %s",
                             base_id, doc_to_header[base_id])
            else:
                # Fallback to original header or empty string
                doc_to_header[base_id] = doc.metadata.get("header", "")
                logger.debug(
                    "No header found for doc %s, using: %s", base_id, doc_to_header[base_id])
        else:
            # Use header from first chunk of same doc
            if base_id in doc_to_header:
                doc.metadata["header"] = doc_to_header[base_id]
                logger.debug("Fixed header for chunk %s: %s",
                             doc.id, doc.metadata["header"])
            else:
                current_header = doc.metadata.get("header", "")
                logger.warning("No header found for doc %s, chunk %d, using: %s",
                               base_id, chunk_index, current_header)
                doc.metadata["header"] = current_header

        fixed_chunked_docs.append(doc)

    # Save chunked documents with fixed headers
    save_file([{
        "id": doc.id,
        "source_url": doc.metadata.get("source_url", ""),
        "doc_index": doc.metadata.get("doc_index", 0),
        "chunk_index": doc.metadata.get("chunk_index", 0),
        "header": doc.metadata.get("header", ""),
        "tokens": doc.metadata.get("tokens", 0),
        "text": doc.text,
    } for doc in fixed_chunked_docs], f"{output_dir}/chunked_docs.json")
    logger.debug("Saved chunked docs to %s/chunked_docs.json", output_dir)

    model_name = "static-retrieval-mrl-en-v1"

    def embed_func(x): return generate_embeddings(
        x, model_name, show_progress=True)
    search_results = search_docs(
        query, fixed_chunked_docs, embed_func, top_k=10,
        header_weight=0.5, content_weight=0.5
    )

    for result in search_results:
        base_id = result['id'].split('_chunk_')[0]
        logger.debug("Searching for original doc with base ID: %s", base_id)
        logger.debug("Available doc IDs: %s", [doc.id for doc in docs])
        try:
            original_doc = next(doc for doc in docs if doc.id == base_id)
            logger.info(
                f"\nRank {result['rank']} (Doc: {result['doc_index']} | Chunk: {result['chunk_index']} | Tokens: {result['tokens']}):")
            print(f"Score: {result['score']:.4f}")
            print(f"Header: {result['header']}")
            print(f"Parent Header: {result['parent_header']}")
            print(f"Chunk:\n{result['text']}")
            print(f"Original Document:\n{original_doc.text}")
            print(
                f"Similarities: text={result['sim_text']:.4f}, header={result['sim_header']:.4f}")
        except StopIteration:
            logger.error(
                f"No original document found for base ID: %s", base_id)
            continue

    save_file([result["text"]
              for result in search_results], f"{output_dir}/results.json")
    save_file(search_results, f"{output_dir}/search_results.json")
    logger.info("Saved search results to %s/results.json", output_dir)
