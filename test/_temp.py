import argparse
import logging
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# Add the parent directory to path to allow importing jet modules
# Adjust this path based on your actual project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from jet.adapters.llama_cpp.config import EMBED_DIMS
    from jet.adapters.llama_cpp.embed_utils import embed, shutdown_embed_pool
except ImportError:
    print(
        "Error: Could not import jet modules. Ensure 'jet_python_modules' is in your path."
    )
    sys.exit(1)

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Lightweight BM25 implementation using pure Python/NumPy.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.corpus = []
        self.documents = []

    def add_documents(self, docs: List[Dict[str, Any]]):
        logger.info("Indexing documents for BM25...")
        self.documents = docs
        self.corpus = [doc["content"].lower().split() for doc in docs]

        # Calculate document frequencies
        df = {}
        N = len(self.corpus)
        for tokens in self.corpus:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1

        self.avgdl = sum(len(tokens) for tokens in self.corpus) / N if N > 0 else 0

        # Calculate IDF
        for token, freq in df.items():
            self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)

        self.doc_len = [len(tokens) for tokens in self.corpus]
        logger.info("BM25 indexing complete.")

    def search(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        query_tokens = query.lower().split()
        scores = np.zeros(len(self.corpus))

        for i, tokens in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_len[i]
            for token in query_tokens:
                if token in self.idf:
                    idf_val = self.idf[token]
                    tf = tokens.count(token)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * (doc_len / self.avgdl)
                    )
                    if denominator != 0:
                        score += idf_val * (numerator / denominator)
            scores[i] = score

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        return results


class VectorRetriever:
    """
    Handles dense vector search using NumPy (Dot Product on Normalized Vectors).
    """

    def __init__(self):
        self.embeddings = None
        self.documents = []

    def add_documents(self, docs: List[Dict[str, Any]], embeddings: np.ndarray):
        logger.info(f"Indexing {len(docs)} documents into Vector Store...")
        self.documents = docs
        self.embeddings = embeddings

        # Normalize embeddings for Cosine Similarity via Dot Product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.embeddings = self.embeddings / norms
        logger.info("Vector indexing complete.")

    def search(
        self, query_embedding: np.ndarray, k: int = 20
    ) -> List[Tuple[int, float]]:
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_vec = query_embedding / query_norm

        # Calculate cosine similarity (dot product of normalized vectors)
        scores = np.dot(self.embeddings, query_vec)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((int(idx), float(scores[idx])))
        return results


class ReciprocalRankFusion:
    @staticmethod
    def fuse(
        results_list: List[List[Tuple[int, float]]], k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.
        """
        doc_scores = {}
        for ranked_list in results_list:
            for rank, (doc_idx, _) in enumerate(ranked_list):
                # RRF Formula: 1 / (k + rank)
                rrf_score = 1.0 / (k + (rank + 1))
                if doc_idx in doc_scores:
                    doc_scores[doc_idx] += rrf_score
                else:
                    doc_scores[doc_idx] = rrf_score
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


class MMRProcessor:
    @staticmethod
    def rerank_with_mmr(
        query_embedding: np.ndarray,
        candidate_indices: List[int],
        all_embeddings: np.ndarray,
        lambda_mult: float = 0.5,
        top_k: int = 5,
    ) -> List[int]:
        """
        Reranks document indices using Maximal Marginal Relevance (MMR).
        """
        if not candidate_indices:
            return []

        k = min(top_k, len(candidate_indices))
        candidate_embeds = all_embeddings[candidate_indices]

        # Cosine similarity to query (embeddings are already normalized)
        sim_to_query = np.dot(candidate_embeds, query_embedding)

        selected_local_indices = []
        remaining_local_indices = list(range(len(candidate_indices)))

        while len(selected_local_indices) < k and remaining_local_indices:
            mmr_scores = {}
            for local_idx in remaining_local_indices:
                relevance_score = sim_to_query[local_idx]

                if selected_local_indices:
                    selected_embeds = candidate_embeds[selected_local_indices]
                    current_doc_embed = candidate_embeds[local_idx]
                    # Cosine similarity between current and selected
                    sims_to_selected = np.dot(selected_embeds, current_doc_embed)
                    diversity_penalty = np.max(sims_to_selected)
                else:
                    diversity_penalty = 0.0

                # MMR Formula: λ * Relevance - (1 - λ) * Diversity Penalty
                mmr_score = (lambda_mult * relevance_score) - (
                    (1 - lambda_mult) * diversity_penalty
                )
                mmr_scores[local_idx] = mmr_score

            # Select best
            best_local_idx = max(mmr_scores, key=mmr_scores.get)
            selected_local_indices.append(best_local_idx)
            remaining_local_indices.remove(best_local_idx)

        # Map back to original global indices
        return [candidate_indices[i] for i in selected_local_indices]


class WorkflowBManager:
    def __init__(self, args):
        self.args = args
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        self.all_embeddings = None

    def ingest_data(self, docs: List[Dict[str, Any]]):
        """Ingest documents and generate embeddings using jet.embed_utils"""
        contents = [doc["content"] for doc in docs]

        # Use jet's robust batch embedding
        logger.info("Generating embeddings using jet.embed_utils...")
        embeddings = embed(
            text=contents,
            return_format="numpy",
            show_progress=True,
            progress_description="Indexing Documents",
        )

        self.all_embeddings = embeddings.astype(np.float32)

        self.vector_retriever.add_documents(docs, self.all_embeddings)
        self.bm25_retriever.add_documents(docs)

    def run(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"--- Starting Workflow B for Query: '{query}' ---")

        # Stage 1: Parallel Retrieval
        logger.info("Stage 1: Hybrid Retrieval...")
        query_emb = embed(text=query, return_format="numpy").astype(np.float32)

        # Normalize query once for all calculations
        query_norm = np.linalg.norm(query_emb)
        query_vec_norm = query_emb / query_norm if query_norm > 0 else query_emb

        vector_results = self.vector_retriever.search(
            query_vec_norm, k=self.args.candidates_limit
        )
        bm25_results = self.bm25_retriever.search(query, k=self.args.candidates_limit)

        # Stage 2: RRF Fusion
        logger.info("Stage 2: Reciprocal Rank Fusion...")
        fused_results = ReciprocalRankFusion.fuse([vector_results, bm25_results], k=60)

        candidate_indices = [
            idx for idx, _ in fused_results[: self.args.candidates_limit]
        ]

        if not candidate_indices:
            return []

        # PRE-FILTER: Strict Relevance Floor
        # We discard anything that isn't at least somewhat semantically related.
        # This prevents "random" docs from entering the MMR pool.
        min_similarity_threshold = 0.25
        valid_candidates = []

        for idx in candidate_indices:
            # Calculate cosine sim against normalized query
            sim = float(np.dot(self.all_embeddings[idx], query_vec_norm))
            if sim >= min_similarity_threshold:
                valid_candidates.append(idx)

        # Fallback: If strict filtering removes everything, take the top 3 regardless
        if len(valid_candidates) < 3:
            logger.warning(
                f"Strict filtering left only {len(valid_candidates)} docs. Relaxing threshold."
            )
            valid_candidates = candidate_indices[:5]

        logger.info(
            f"Candidate Pool: {len(candidate_indices)} -> {len(valid_candidates)} (after sim >= {min_similarity_threshold})"
        )

        # Stage 3: MMR Reranking
        logger.info(
            f"Stage 3: MMR Reranking (lambda={self.args.lambda_val}, top_k={self.args.top_k})..."
        )

        mmr_ranked_indices = MMRProcessor.rerank_with_mmr(
            query_embedding=query_vec_norm,
            candidate_indices=valid_candidates,
            all_embeddings=self.all_embeddings,
            lambda_mult=self.args.lambda_val,
            top_k=self.args.top_k,
        )

        # Extract detailed scores for display
        final_results = []
        for rank, global_idx in enumerate(mmr_ranked_indices, 1):
            doc = self.vector_retriever.documents[global_idx]

            cos_sim = float(np.dot(self.all_embeddings[global_idx], query_vec_norm))
            bm25_score = next(
                (score for idx, score in bm25_results if idx == global_idx), 0.0
            )
            rrf_score = next(
                (score for idx, score in fused_results if idx == global_idx), 0.0
            )

            final_results.append(
                {
                    "document": doc,
                    "cosine_similarity": round(cos_sim, 4),
                    "bm25_score": round(bm25_score, 4),
                    "rrf_score": round(rrf_score, 4),
                    "mmr_rank": rank,
                }
            )

        return final_results


def get_sample_data():
    """Generates sample data for demonstration."""
    docs = []
    contents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast red fox leaps above a sleeping canine.",  # Similar to 0
        "Python is a popular programming language for AI.",
        "Java is widely used for enterprise applications.",
        "Machine learning models require large datasets.",
        "Deep learning is a subset of machine learning.",  # Similar to 4
        "The stock market crashed due to economic instability.",
        "Financial markets experienced a significant downturn.",  # Similar to 6
        "Climate change is causing global temperature rises.",
        "Renewable energy sources are essential for sustainability.",
        "Black jackets are fashionable in winter.",
        "A dark coat is suitable for cold weather.",  # Similar to 10
        "Hoodies provide casual comfort and warmth.",
        "Blazers are appropriate for professional settings.",
        "Windbreakers protect against rain and wind.",
    ]

    for i, content in enumerate(contents):
        docs.append(
            {"id": f"doc_{i}", "content": content, "metadata": {"source": "sample"}}
        )

    return docs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced Hybrid Search + MMR Pipeline (Workflow B)"
    )

    parser.add_argument(
        "-l",
        "--lambda-val",
        type=float,
        default=0.5,
        help="MMR Lambda (0.0=diversity, 1.0=relevance)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of final documents to return",
    )
    parser.add_argument(
        "-c",
        "--candidates_limit",
        type=int,
        default=20,
        help="Number of initial candidates from Hybrid Search",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="black jacket",
        help="Search query",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Loading sample data...")
    sample_docs = get_sample_data()

    manager = WorkflowBManager(args)
    manager.ingest_data(sample_docs)

    results = manager.run(args.query)

    print("\n" + "=" * 90)
    print(f"RESULTS FOR QUERY: '{args.query}'")
    print(f"Strategy: Hybrid (BM25+Vector) -> RRF -> MMR (λ={args.lambda_val})")
    print("=" * 90)

    if results:
        # Header
        print(f"{'Rank':<5} | {'Sim':<6} | {'BM25':<8} | {'RRF':<8} | Content Preview")
        print("-" * 90)

        for res in results:
            doc = res["document"]
            # Truncate content for clean display
            preview = (
                doc["content"][:50] + "..."
                if len(doc["content"]) > 50
                else doc["content"]
            )

            print(
                f"{res['mmr_rank']:<5} | "
                f"{res['cosine_similarity']:<6.3f} | "
                f"{res['bm25_score']:<8.3f} | "
                f"{res['rrf_score']:<8.4f} | "
                f"{preview}"
            )
    else:
        print("No results found.")

    print("\n" + "=" * 90)

    # Gracefully shutdown the embedding pool
    shutdown_embed_pool()


if __name__ == "__main__":
    main()
