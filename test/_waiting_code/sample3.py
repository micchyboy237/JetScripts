import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import torch
from concurrent.futures import ThreadPoolExecutor
import re
import logging
from tqdm import tqdm
from jet.file.utils import load_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load SearxNG-scraped data


def load_documents(file_path):
    logger.info("Loading documents from %s", file_path)
    docs = load_file(file_path)
    documents = [
        "\n".join([
            doc["metadata"].get("parent_header") or "",
            doc["metadata"]["header"],
            doc["metadata"]["content"]
        ]).strip()
        for doc in tqdm(docs, desc="Processing documents")
        if doc["metadata"]["header_level"] != 1
    ]
    return documents

# Split documents


def split_document(text, chunk_size=800, overlap=200):
    logger.info("Splitting document into chunks")
    chunks = []
    headers = []
    lines = text.split("\n")
    current_chunk = ""
    current_len = 0
    for line in lines:
        if line.startswith(("#", "##", "###")):
            headers.append(line)
        line_len = len(line.split())
        if line.startswith(("#", "##", "###")) or current_len + line_len > chunk_size:
            if current_chunk:
                chunks.append({"text": current_chunk.strip(),
                              "headers": headers.copy()})
            if line.startswith(("#", "##", "###")):
                current_chunk = line
                current_len = line_len
            else:
                current_chunk = line
                current_len = line_len
        else:
            current_chunk += "\n" + line
            current_len += line_len
    if current_chunk:
        chunks.append({"text": current_chunk.strip(),
                      "headers": headers.copy()})
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

# Main RAG pipeline


def main():
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    query = "best RAG techniques for web data"

    # Load documents
    documents = load_documents(docs_file)

    # Split documents into chunks
    logger.info("Splitting %d documents into chunks", len(documents))
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
        chunks.extend(split_document(doc))

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
    initial_docs = [filtered_chunks[i] for i in indices[0]]

    # Cross-encoder reranking
    logger.info("Reranking %d documents with cross-encoder", len(initial_docs))
    batch_size = 8
    pairs = [[query, doc["text"]] for doc in initial_docs]
    scores = []
    try:
        for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking batches"):
            batch = pairs[i:i+batch_size]
            scores.extend(cross_encoder.predict(batch))
    except Exception as e:
        logger.error("Error in reranking: %s", e)
        scores = [0] * len(pairs)  # Fallback
    reranked_indices = np.argsort(scores)[::-1][:5]
    reranked_docs = [initial_docs[i] for i in reranked_indices]

    # Output results
    logger.info("Outputting top 5 reranked documents")
    for i, doc in enumerate(reranked_docs):
        print(f"Rank {i+1}: {doc['text'][:200]}...")
        print(f"Headers: {doc['headers']}")


if __name__ == "__main__":
    main()
