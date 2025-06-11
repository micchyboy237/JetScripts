import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from collections import defaultdict
from math import log
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
from jet.file.utils import load_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom BM25 implementation


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        logger.info(
            "Computing average document length and IDF for %d documents", len(documents))
        self.avgdl = sum(len(doc.split()) for doc in tqdm(
            documents, desc="Calculating avgdl")) / len(documents)
        self.idf = self._compute_idf()

    def _compute_idf(self):
        logger.info("Computing IDF scores")
        idf = {}
        N = len(self.documents)
        for doc in tqdm(self.documents, desc="Processing documents for IDF"):
            for term in set(doc.split()):
                idf[term] = idf.get(term, 0) + 1
        return {term: log((N - freq + 0.5) / (freq + 0.5) + 1) for term, freq in idf.items()}

    def score(self, query):
        logger.info("Scoring documents with BM25 for query: %s", query)
        scores = []
        query_terms = query.split()
        for doc in tqdm(self.documents, desc="Scoring documents"):
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

# Split documents with header preservation


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

# Load documents


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

# Main RAG pipeline


def main():
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    query = "best RAG techniques for web data"
    cache_file = "query_cache.pkl"

    # Load documents
    documents = load_documents(docs_file)

    # Split documents into chunks
    logger.info("Splitting %d documents into chunks", len(documents))
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
        chunks.extend(split_document(doc))
    chunk_texts = [chunk["text"] for chunk in chunks]
    logger.info("Generated %d chunks", len(chunks))

    # Initialize models
    logger.info("Initializing SentenceTransformer and CrossEncoder models")
    global embedder, cross_encoder
    embedder = SentenceTransformer("all-MiniLM-L12-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    # Cache setup
    logger.info("Checking for cached query results")
    cache = Path(cache_file).exists() and pickle.load(
        open(cache_file, "rb")) or {}

    # Check cache
    if query in cache:
        logger.info("Found cached results for query")
        reranked_docs = cache[query]
    else:
        # Dynamic weighting based on query length
        query_len = len(query.split())
        bm25_weight = 0.6 if query_len < 5 else 0.4
        dense_weight = 1 - bm25_weight
        logger.info("Using BM25 weight: %.2f, Dense weight: %.2f",
                    bm25_weight, dense_weight)

        # BM25 retrieval
        bm25 = BM25(chunk_texts)
        bm25_scores = bm25.score(query)
        bm25_top_k = np.argsort(bm25_scores)[-10:][::-1]
        bm25_docs = [chunks[i] for i in bm25_top_k]
        bm25_scores = [bm25_scores[i] for i in bm25_top_k]
        logger.info("Retrieved %d documents with BM25", len(bm25_docs))

        # Dense retrieval with HNSW
        logger.info("Encoding %d chunks for dense retrieval", len(chunk_texts))
        chunk_embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)
        dim = chunk_embeddings.shape[1]
        logger.info("Building HNSW index with dimension %d", dim)
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 40
        index.add(chunk_embeddings)
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        logger.info("Performing HNSW search for top 10 documents")
        distances, indices = index.search(query_embedding, 10)
        dense_docs = [chunks[i] for i in indices[0]]
        dense_scores = [1 / (1 + d) for d in distances[0]]

        # Combine and normalize scores
        logger.info("Combining BM25 and dense retrieval scores")
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
        top_chunks = [
            chunk for chunk in chunks if chunk["text"] in top_doc_texts]
        logger.info("Selected %d top candidates for reranking",
                    len(top_chunks))

        # Batched cross-encoder reranking
        logger.info("Reranking %d documents with cross-encoder",
                    len(top_chunks))
        batch_size = 8
        pairs = [[query, chunk["text"]] for chunk in top_chunks]
        scores = []
        try:
            for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking batches"):
                batch = pairs[i:i + batch_size]
                scores.extend(cross_encoder.predict(batch))
        except Exception as e:
            logger.error("Error in reranking: %s", e)
            scores = [0] * len(pairs)  # Fallback
        reranked_indices = np.argsort(scores)[::-1][:5]
        reranked_docs = [top_chunks[i] for i in reranked_indices]

        # Cache results
        logger.info("Caching results for query")
        cache[query] = reranked_docs
        pickle.dump(cache, open(cache_file, "wb"))

    # Output results
    logger.info("Outputting top 5 reranked documents")
    for i, doc in enumerate(reranked_docs):
        print(f"Rank {i+1}: {doc['text'][:200]}...")
        print(f"Headers: {doc['headers']}")


if __name__ == "__main__":
    main()
