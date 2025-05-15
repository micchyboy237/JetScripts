import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from collections import defaultdict
from math import log
import pickle
from pathlib import Path

# Custom BM25 implementation


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc.split())
                         for doc in documents) / len(documents)
        self.idf = self._compute_idf()

    def _compute_idf(self):
        idf = {}
        N = len(self.documents)
        for doc in self.documents:
            for term in set(doc.split()):
                idf[term] = idf.get(term, 0) + 1
        return {term: log((N - freq + 0.5) / (freq + 0.5) + 1) for term, freq in idf.items()}

    def score(self, query):
        scores = []
        query_terms = query.split()
        for doc in self.documents:
            score = 0
            doc_len = len(doc.split())
            term_freq = defaultdict(int)
            for term in doc.split():
                term_freq[term] += 1
            for term in query_terms:
                if term in self.idf:
                    tf = term_freq[term]
                    score += self.idf[term] * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            scores.append(score)
        return scores


# Load SearxNG-scraped data
documents = []
for file in os.listdir("searxng_data"):
    if file.endswith(".txt"):
        with open(os.path.join("searxng_data", file), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Split documents with header preservation


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
                chunk_text = "\n".join(
                    headers[-1:]) + "\n" + current_chunk.strip() if headers else current_chunk.strip()
                chunks.append({"text": chunk_text, "headers": headers.copy()})
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
        chunk_text = "\n".join(
            headers[-1:]) + "\n" + current_chunk.strip() if headers else current_chunk.strip()
        chunks.append({"text": chunk_text, "headers": headers.copy()})
    return chunks


chunks = []
for doc in documents:
    chunks.extend(split_document(doc))
chunk_texts = [chunk["text"] for chunk in chunks]

# Initialize models
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Cache setup
cache_file = "query_cache.pkl"
cache = Path(cache_file).exists() and pickle.load(open(cache_file, "rb")) or {}

# BM25 retrieval
bm25 = BM25(chunk_texts)
query = "best RAG techniques for web data"

# Check cache
if query in cache:
    reranked_docs = cache[query]
else:
    # Dynamic weighting based on query length
    query_len = len(query.split())
    bm25_weight = 0.6 if query_len < 5 else 0.4
    dense_weight = 1 - bm25_weight

    # BM25 scores
    bm25_scores = bm25.score(query)
    bm25_top_k = np.argsort(bm25_scores)[-10:][::-1]
    bm25_docs = [chunks[i] for i in bm25_top_k]
    bm25_scores = [bm25_scores[i] for i in bm25_top_k]

    # Dense retrieval with HNSW
    chunk_embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)  # HNSW with 32 neighbors
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 40
    index.add(chunk_embeddings)
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 10)
    dense_docs = [chunks[i] for i in indices[0]]
    dense_scores = [1 / (1 + d)
                    for d in distances[0]]  # Convert distance to score

    # Combine and normalize scores
    combined_docs = list(
        set([d["text"] for d in bm25_docs] + [d["text"] for d in dense_docs]))
    doc_scores = {}
    for i, doc in enumerate(bm25_docs):
        doc_scores[doc["text"]] = doc_scores.get(
            doc["text"], 0) + bm25_weight * bm25_scores[i]
    for i, doc in enumerate(dense_docs):
        doc_scores[doc["text"]] = doc_scores.get(
            doc["text"], 0) + dense_weight * dense_scores[i]

    # Select top candidates
    top_docs = sorted(doc_scores.items(),
                      key=lambda x: x[1], reverse=True)[:20]
    top_doc_texts = [doc for doc, _ in top_docs]
    top_chunks = [chunk for chunk in chunks if chunk["text"] in top_doc_texts]

    # Batched cross-encoder reranking
    batch_size = 8
    pairs = [[query, chunk["text"]] for chunk in top_chunks]
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        scores.extend(cross_encoder.predict(batch))
    reranked_indices = np.argsort(scores)[::-1][:5]
    reranked_docs = [top_chunks[i] for i in reranked_indices]

    # Cache results
    cache[query] = reranked_docs
    pickle.dump(cache, open(cache_file, "wb"))

# Output results
for i, doc in enumerate(reranked_docs):
    print(f"Rank {i+1}: {doc['text'][:200]}...")
    print(f"Headers: {doc['headers']}")
