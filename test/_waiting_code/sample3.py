import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import torch
from concurrent.futures import ThreadPoolExecutor
import re
from tqdm import tqdm
from jet.file.utils import load_file
from jet.logger import logger


# Load SearxNG-scraped data


def load_documents(file_path):
    logger.info("Loading documents from %s", file_path)
    docs = load_file(file_path)
    documents = [
        {
            "text": "\n".join([
                doc["metadata"].get("parent_header") or "",
                doc["metadata"]["header"],
                doc["metadata"]["content"]
            ]).strip(),
            "id": idx
        }
        for idx, doc in enumerate(tqdm(docs, desc="Processing documents"))
        if doc["metadata"]["header_level"] != 1
    ]
    return documents

# Split documents


def split_document(doc_text, doc_id, chunk_size=800, overlap=200):
    logger.info("Splitting document ID %d into chunks", doc_id)
    chunks = []
    headers = []
    lines = doc_text.split("\n")
    current_chunk = ""
    current_len = 0
    for line in lines:
        if line.startswith(("# ", "## ")):
            headers.append(line)
        line_len = len(line.split())
        if line.startswith(("# ", "## ")) or current_len + line_len > chunk_size:
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "headers": headers.copy(),
                    "doc_id": doc_id
                })
            if line.startswith(("# ", "## ")):
                current_chunk = line
                current_len = line_len
            else:
                current_chunk = line
                current_len = line_len
        else:
            current_chunk += "\n" + line
            current_len += line_len
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "headers": headers.copy(),
            "doc_id": doc_id
        })
    return chunks

# Filter chunks by headers


def filter_by_headers(chunks, query):
    logger.info("Filtering chunks by headers for query: %s", query)
    query_terms = set(query.lower().split())
    filtered = []
    for chunk in tqdm(chunks, desc="Filtering chunks"):
        headers = [h.lower() for h in chunk["headers"]]
        if any(any(term in h for term in query_terms) for h in headers) or not headers:
            filtered.append(chunk)
    return filtered if filtered else chunks

# Embed chunks in parallel


def embed_chunk(chunk):
    return embedder.encode(chunk, convert_to_numpy=True)


def embed_chunks_parallel(chunk_texts):
    logger.info("Embedding %d chunks in parallel", len(chunk_texts))
    with ThreadPoolExecutor() as executor:
        embeddings = list(tqdm(
            executor.map(embed_chunk, chunk_texts),
            total=len(chunk_texts),
            desc="Embedding chunks"
        ))
    return np.vstack(embeddings)

# Reconstruct original document from chunk


def get_original_document(doc_id, documents):
    logger.info("Retrieving original document for ID %d", doc_id)
    for doc in documents:
        if doc["id"] == doc_id:
            return doc["text"]
    return None

# Main RAG pipeline


def main():
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    query = "List all ongoing and upcoming isekai anime 2025."

    # Load documents
    documents = load_documents(docs_file)

    # Split documents into chunks
    logger.info("Splitting %d documents into chunks", len(documents))
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
        chunks.extend(split_document(doc["text"], doc["id"]))

    # Filter chunks
    filtered_chunks = filter_by_headers(chunks, query)
    chunk_texts = [chunk["text"] for chunk in filtered_chunks]
    logger.info("Filtered to %d chunks", len(chunk_texts))

    # Initialize models
    logger.info("Initializing SentenceTransformer and CrossEncoder models")
    global embedder, cross_encoder
    embedder = SentenceTransformer("all-MiniLM-L12-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # # Quantize cross-encoder
    # logger.info("Quantizing cross-encoder model")
    # cross_encoder.model.eval()
    # with torch.no_grad():
    #     cross_encoder.model = torch.quantization.quantize_dynamic(
    #         cross_encoder.model, {torch.nn.Linear}, dtype=torch.qint8
    #     )

    # Embed chunks
    chunk_embeddings = embed_chunks_parallel(chunk_texts)

    # FAISS index
    logger.info("Building FAISS index")
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embeddings)

    # Initial retrieval
    k = 20 if len(chunk_texts) < 1000 else 50
    logger.info("Performing FAISS search with top-k=%d", k)
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    # Convert distances to similarity scores
    embed_scores = [1 / (1 + d) for d in distances[0]]
    initial_docs = [
        (filtered_chunks[i], embed_scores[j])
        for j, i in enumerate(indices[0])
    ]
    logger.debug("FAISS search results: indices=%s, embed_scores=%s",
                 indices[0][:5], embed_scores[:5])

    # Cross-encoder reranking
    logger.info("Reranking %d documents with cross-encoder", len(initial_docs))
    batch_size = 8
    pairs = [[query, doc["text"]] for doc, _ in initial_docs]
    scores = []
    try:
        for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking batches"):
            batch = pairs[i:i + batch_size]
            batch_scores = cross_encoder.predict(batch)
            scores.extend(batch_scores)
            logger.debug("Batch %d rerank scores: %s",
                         i // batch_size, batch_scores)
    except Exception as e:
        logger.error("Error in reranking: %s", e)
        scores = [0] * len(pairs)  # Fallback
    logger.debug("All rerank scores: %s", scores[:10])
    reranked_indices = np.argsort(scores)[::-1][:10]
    reranked_docs = [
        (initial_docs[i][0], initial_docs[i][1], scores[i])
        for i in reranked_indices
    ]
    logger.debug("Top 5 reranked docs: %s", [
                 (doc[0]["doc_id"], doc[1], doc[2]) for doc in reranked_docs])

    # Output results with original documents and scores
    logger.info(
        "Outputting top 5 reranked documents with original source and scores")
    seen_doc_ids = set()  # Track unique documents to avoid duplicates
    for i, (chunk, embed_score, rerank_score) in enumerate(reranked_docs):
        doc_id = chunk["doc_id"]
        if doc_id not in seen_doc_ids:
            original_doc = get_original_document(doc_id, documents)
            if original_doc:
                print(f"Rank {i+1} (Document ID {doc_id}):")
                print(f"Embedding Score: {embed_score:.4f}")
                print(f"Rerank Score: {rerank_score:.4f}")
                print(f"Chunk Preview: {chunk['text'][:200]}...")
                print(f"Headers: {chunk['headers']}")
                print(f"Original Document:\n{original_doc}\n")
                seen_doc_ids.add(doc_id)
            else:
                logger.warning("Original document not found for ID %d", doc_id)
                print(f"Rank {i+1} (Document ID {doc_id}):")
                print(f"Embedding Score: {embed_score:.4f}")
                print(f"Rerank Score: {rerank_score:.4f}")
                print(f"Chunk Preview: {chunk['text'][:200]}...")
                print(f"Headers: {chunk['headers']}")
                print("Original Document: Not found\n")


if __name__ == "__main__":
    main()
