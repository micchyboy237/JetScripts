from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI
import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class ScoredItem:
    """Simple container for a retrieved item + score"""

    item: Any
    score: float

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ScoredItem):
            return NotImplemented
        return self.score < other.score  # higher score = better


class LlamaCppEmbedder:
    """Thin wrapper around openai client for llama.cpp embedding server"""

    def __init__(
        self,
        base_url: str = "http://shawn-pc.local:8081/v1",
        model: str = "nomic-embed-text-v1.5",
    ):
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed",  # llama.cpp ignores it
            timeout=60.0,
        )

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        prefix = "search_query: " if is_query else "search_document: "
        prefixed = [prefix + t.strip() for t in texts if t.strip()]
        if not prefixed:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=prefixed,
            encoding_format="float",
            # dimensions=512,  # ← try this too (Matryoshka), saves storage & often similar perf
        )
        return [d.embedding for d in response.data]


class VectorRetriever:
    """Simple in-memory vector store + dense retriever"""

    def __init__(self, embedder: LlamaCppEmbedder):
        self.embedder = embedder
        self.documents: List[Any] = []
        self.embeddings: NDArray[np.float32] = np.empty((0, 0), dtype=np.float32)

    def index(self, documents: List[Any], texts: List[str]):
        """Index documents with their corresponding texts"""
        if len(documents) != len(texts):
            raise ValueError("documents and texts must have same length")

        if not texts:
            return

        embeddings = self.embedder.embed(texts)
        self.documents = self.documents + documents
        self.embeddings = (
            np.vstack([self.embeddings, np.array(embeddings, dtype=np.float32)])
            if self.embeddings.size > 0
            else np.array(embeddings, dtype=np.float32)
        )

    def search(self, query: str, k: int = 50) -> List[ScoredItem]:
        if self.embeddings.size == 0:
            return []

        [q_emb] = self.embedder.embed([query], is_query=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        q_emb /= np.linalg.norm(q_emb) + 1e-10  # normalize

        scores = self.embeddings @ q_emb
        top_indices = np.argsort(scores)[-k:][::-1]

        return [ScoredItem(self.documents[i], float(scores[i])) for i in top_indices]


class BM25Retriever:
    """BM25 retriever using rank_bm25"""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Any] = []

    def index(self, documents: List[Any], texts: List[str]):
        if len(documents) != len(texts):
            raise ValueError("documents and texts must have same length")

        tokenized = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.documents.extend(documents)

    def search(self, query: str, k: int = 50) -> List[ScoredItem]:
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]

        return [
            ScoredItem(self.documents[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


@dataclass
class HybridConfig:
    k_candidates: int = 80
    k_final: int = 20
    rrf_constant: float = 60.0
    bm25_weight: float = 1.0
    vector_weight: float = 1.0


def reciprocal_rank_fusion(
    result_lists: Sequence[List[ScoredItem]],
    rrf_k: float = 60.0,
    weights: Optional[List[float]] = None,
) -> List[ScoredItem]:
    """Merge ranked lists using Reciprocal Rank Fusion"""
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError("weights length must match number of lists")

    score_map: Dict[str, float] = {}  # doc_id → fused RRF score
    id_to_doc: Dict[str, Any] = {}  # doc_id → full document object

    for ranked_list, weight in zip(result_lists, weights):
        for rank, res in enumerate(ranked_list, 1):
            doc_id = res.item["id"]  # assumes every doc has unique "id"
            rrf_score = weight / (rrf_k + rank)
            score_map[doc_id] = score_map.get(doc_id, 0.0) + rrf_score
            id_to_doc[doc_id] = res.item

    fused = [
        ScoredItem(id_to_doc[doc_id], score)
        for doc_id, score in sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    ]
    return fused


class HybridSearcher:
    """Hybrid BM25 + dense vector search with RRF fusion"""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        config: HybridConfig = HybridConfig(),
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.config = config

    def search(self, query: str) -> List[ScoredItem]:
        bm25_results = self.bm25.search(query, k=self.config.k_candidates)
        vector_results = self.vector.search(query, k=self.config.k_candidates)

        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            rrf_k=self.config.rrf_constant,
            weights=[self.config.bm25_weight, self.config.vector_weight],
        )

        return fused[: self.config.k_final]


# ────────────────────────────────────────────────
#                   Example Usage
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # Prepare data
    docs = [
        {
            "id": "d1",
            "title": "Hybrid vector search best practices 2025",
            "content": "Use RRF for combining BM25 and dense embeddings...",
        },
        {
            "id": "d2",
            "title": "nomic-embed-text-v1.5 performance",
            "content": "Very fast on llama.cpp especially with Q5_K_M quantization",
        },
        {
            "id": "d3",
            "title": "Reciprocal Rank Fusion",
            "content": "Simple yet powerful fusion method used in Elastic and Weaviate",
        },
        {
            "id": "d4",
            "title": "Local embedding servers",
            "content": "llama.cpp provides OpenAI compatible API for embedding models",
        },
        {
            "id": "d5",
            "title": "BM25 is still very strong",
            "content": "Especially good at rare terms, IDs, exact matches",
        },
    ]

    texts = [d["title"] + " " + d["content"] for d in docs]

    # Initialize components
    embedder = LlamaCppEmbedder(
        base_url="http://shawn-pc.local:8081/v1", model="nomic-embed-text-v1.5"
    )

    bm25_ret = BM25Retriever()
    vector_ret = VectorRetriever(embedder)

    bm25_ret.index(docs, texts)
    vector_ret.index(docs, [d["title"] + " " + d["content"] for d in docs])

    hybrid = HybridSearcher(
        bm25_ret,
        vector_ret,
        HybridConfig(
            k_candidates=10,
            k_final=5,
            bm25_weight=1.2,  # slightly favor keyword matches
            vector_weight=1.0,
        ),
    )

    # Query
    query = "fast local embeddings with llama.cpp"
    results = hybrid.search(query)

    print(f"\nResults for: {query!r}\n")
    for i, res in enumerate(results, 1):
        doc = res.item
        print(f"{i:2d}. {res.score:6.4f}  {doc['id']}  {doc['title']}")
