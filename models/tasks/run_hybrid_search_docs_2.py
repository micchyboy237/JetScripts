from typing import Union, List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Callable, Union, TypedDict
from jet.vectors.document_types import HeaderDocument
from jet.token.token_utils import split_headers
from jet.file.utils import load_file, save_file
import os
from unittest.mock import Mock
import pytest
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from jet.vectors.document_types import HeaderDocument, HeaderMetadata
from jet.logger import logger


def chunk_headers(docs: List[HeaderDocument], max_tokens: int = 500) -> List[HeaderDocument]:
    """
    Chunk documents into smaller HeaderDocument objects with generated headers.

    Args:
        docs: List of HeaderDocument objects to chunk.
        max_tokens: Maximum token count per chunk (approximated by word count).

    Returns:
        List of chunked HeaderDocument objects with headers and metadata.
    """
    logger.debug("Starting chunk_headers with %d documents", len(docs))
    chunked_docs: List[HeaderDocument] = []
    chunk_index = 0

    for doc in docs:
        metadata = HeaderMetadata(**doc.metadata)
        text_lines = metadata.get("texts", doc.text.splitlines())
        current_chunk = []
        current_tokens = 0
        parent_header = metadata.get("parent_header", "")
        doc_index = metadata.get("doc_index", 0)

        for line in text_lines:
            # Approximate token count (1 word ≈ 1.3 tokens)
            line_tokens = len(line.split()) * 1.3
            if current_tokens + line_tokens > max_tokens and current_chunk:
                # Create a new chunk
                chunk_text = "\n".join(current_chunk)
                header = current_chunk[0][:100] + \
                    "..." if current_chunk else ""
                chunked_docs.append(HeaderDocument(
                    id=f"{doc.id}_chunk_{chunk_index}",
                    text=chunk_text,
                    metadata={
                        "header": header,
                        "parent_header": parent_header,
                        "header_level": metadata.get("header_level", 0) + 1,
                        "content": chunk_text,
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                        "texts": current_chunk
                    }
                ))
                logger.debug("Created chunk %d for doc %s: header=%s",
                             chunk_index, doc.id, header)
                chunk_index += 1
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            header = current_chunk[0][:100] + "..." if current_chunk else ""
            chunked_docs.append(HeaderDocument(
                id=f"{doc.id}_chunk_{chunk_index}",
                text=chunk_text,
                metadata={
                    "header": header,
                    "parent_header": parent_header,
                    "header_level": metadata.get("header_level", 0) + 1,
                    "content": chunk_text,
                    "doc_index": doc_index,
                    "chunk_index": chunk_index,
                    "texts": current_chunk
                }
            ))
            logger.debug("Created final chunk %d for doc %s: header=%s",
                         chunk_index, doc.id, header)
            chunk_index += 1

    logger.info("Generated %d chunks from %d documents",
                len(chunked_docs), len(docs))
    return chunked_docs


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


class TestChunkHeaders:
    def test_chunk_headers_single_doc(self):
        doc = HeaderDocument(
            id="doc1",
            text="Line 1\nLine 2\nLine 3\nLine 4",
            metadata={"doc_index": 1, "parent_header": "Parent"}
        )
        # Small max_tokens for testing
        result = chunk_headers([doc], max_tokens=2)
        expected = [
            HeaderDocument(
                id="doc1_chunk_0",
                text="Line 1\nLine 2",
                metadata={
                    "header": "Line 1...",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "Line 1\nLine 2",
                    "doc_index": 1,
                    "chunk_index": 0,
                    "texts": ["Line 1", "Line 2"]
                }
            ),
            HeaderDocument(
                id="doc1_chunk_1",
                text="Line 3\nLine 4",
                metadata={
                    "header": "Line 3...",
                    "parent_header": "Parent",
                    "header_level": 1,
                    "content": "Line 3\nLine 4",
                    "doc_index": 1,
                    "chunk_index": 1,
                    "texts": ["Line 3", "Line 4"]
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.id == e.id
            assert r.text == e.text
            assert r.metadata == e.metadata
        logger.debug("Test chunk_headers_single_doc passed")


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


def generate_embeddings(
    input_data: Union[str, List[str]],
    model: str = "static-retrieval-mrl-en-v1",
    batch_size: int = 32,
    show_progress: bool = False
) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings for a single string or list of strings using SentenceTransformer.

    Args:
        input_data: A single string or list of strings to embed.
        model: Name of the SentenceTransformer model to use.
        batch_size: Batch size for embedding multiple strings.
        show_progress: Whether to display a progress bar for batch processing.

    Returns:
        List[float] for a single string input, or List[List[float]] for a list of strings.
    """
    logger.info("Generating embeddings for input type: %s, model: %s, show_progress: %s",
                type(input_data), model, show_progress)

    try:
        # Initialize SentenceTransformer with ONNX backend for Mac M1 compatibility
        embedder = SentenceTransformer(model, device="cpu", backend="onnx")
        logger.debug(
            "Embedding model initialized with device: %s", embedder.device)

        if isinstance(input_data, str):
            # Handle single string input
            logger.debug("Processing single string input: %s", input_data[:50])
            embedding = embedder.encode(input_data, convert_to_numpy=True)
            embedding = np.ascontiguousarray(embedding.astype(np.float32))
            logger.debug("Generated embedding shape: %s", embedding.shape)
            return embedding.tolist()

        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # Handle list of strings input
            logger.debug("Processing %d strings in batches of %d",
                         len(input_data), batch_size)
            if not input_data:
                logger.info(
                    "Empty input list, returning empty list of embeddings")
                return []

            embeddings = []
            # Use tqdm for progress bar if show_progress is True
            iterator = range(0, len(input_data), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Embedding texts", total=len(
                    range(0, len(input_data), batch_size)))

            for i in iterator:
                batch = input_data[i:i + batch_size]
                logger.debug("Encoding batch %d-%d", i,
                             min(i + batch_size, len(input_data)))
                batch_embeddings = embedder.encode(
                    batch,
                    convert_to_numpy=True,
                    batch_size=batch_size
                )
                batch_embeddings = np.ascontiguousarray(
                    batch_embeddings.astype(np.float32))
                embeddings.extend(batch_embeddings.tolist())

            logger.debug("Generated embeddings shape: (%d, %d)", len(
                embeddings), len(embeddings[0]) if embeddings else 0)
            return embeddings

        else:
            logger.error(
                "Invalid input type: %s, expected str or List[str]", type(input_data))
            raise ValueError("Input must be a string or a list of strings")

    except Exception as e:
        logger.error("Failed to generate embeddings: %s", str(e))
        raise


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
    chunked_docs = chunk_headers(docs, max_tokens=500)
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
        logger.success(
            f"\nRank {result['rank']} (Doc: {result['doc_index']} | Chunk: {result['chunk_index']}):")
        print(f"Score: {result['score']:.4f}")
        print(f"Header: {result['header']}")
        print(f"Parent Header: {result['parent_header']}")
        print(f"Original Document:\n{result['text']}")

    save_file(search_results, f"{output_dir}/results.json")
    logger.info("Saved search results to %s/results.json", output_dir)
