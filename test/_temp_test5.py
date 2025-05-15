import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import torch
from concurrent.futures import ThreadPoolExecutor
import re

# Load SearxNG-scraped data
documents = []
for file in os.listdir("searxng_data"):
    if file.endswith(".txt"):
        try:
            with open(os.path.join("searxng_data", file), "r", encoding="utf-8") as f:
                documents.append(f.read())
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Split documents


def split_document(text, chunk_size=800, overlap=200):
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


chunks = []
for doc in documents:
    chunks.extend(split_document(doc))

# Pre-filter by header keywords


def filter_by_headers(chunks, query):
    query_terms = set(query.lower().split())
    filtered = []
    for chunk in chunks:
        headers = [h.lower() for h in chunk["headers"]]
        if any(any(term in h for term in query_terms) for h in headers) or not headers:
            filtered.append(chunk)
    return filtered if filtered else chunks  # Fallback to all chunks if none match


query = "best RAG techniques for web data"
filtered_chunks = filter_by_headers(chunks, query)
chunk_texts = [chunk["text"] for chunk in filtered_chunks]

# Initialize models
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Quantize cross-encoder (simplified, requires ONNX export in practice)
cross_encoder.model.eval()
with torch.no_grad():
    cross_encoder.model = torch.quantization.quantize_dynamic(
        cross_encoder.model, {torch.nn.Linear}, dtype=torch.qint8
    )

# Parallel embedding


def embed_chunk(chunk):
    return embedder.encode(chunk, convert_to_numpy=True)


with ThreadPoolExecutor() as executor:
    chunk_embeddings = np.vstack(list(executor.map(embed_chunk, chunk_texts)))

# FAISS for initial retrieval
dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(chunk_embeddings)

# Adaptive top-k
k = 20 if len(chunk_texts) < 1000 else 50
query_embedding = embedder.encode([query], convert_to_numpy=True)
distances, indices = index.search(query_embedding, k)
initial_docs = [filtered_chunks[i] for i in indices[0]]

# Batched cross-encoder reranking
batch_size = 8
pairs = [[query, doc["text"]] for doc in initial_docs]
scores = []
try:
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        scores.extend(cross_encoder.predict(batch))
except Exception as e:
    print(f"Error in reranking: {e}")
    scores = [0] * len(pairs)  # Fallback
reranked_indices = np.argsort(scores)[::-1][:5]
reranked_docs = [initial_docs[i] for i in reranked_indices]

# Output results
for i, doc in enumerate(reranked_docs):
    print(f"Rank {i+1}: {doc['text'][:200]}...")
    print(f"Headers: {doc['headers']}")
