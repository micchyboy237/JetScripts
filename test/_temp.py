from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# =============================================================================
# STEP 1: Sample Job Postings Corpus
# =============================================================================
JOB_POSTINGS = {
    "job_001": {
        "title": "Senior DevOps Engineer",
        "tech_stack": ["Kubernetes", "AWS", "Terraform", "Python"],
        "description": "Manage cloud infrastructure and CI/CD pipelines for microservices.",
        "posted_days_ago": 2,
    },
    "job_002": {
        "title": "Cloud Infrastructure Lead",
        "tech_stack": ["GCP", "Docker", "Ansible"],
        "description": "Lead team responsible for Kubernetes orchestration and AWS migration projects.",
        "posted_days_ago": 5,
    },
    "job_003": {
        "title": "Site Reliability Engineer",
        "tech_stack": ["Kubernetes", "AWS", "Go", "Prometheus"],
        "description": "Ensure reliability and scalability of production systems using SRE best practices.",
        "posted_days_ago": 1,
    },
    "job_004": {
        "title": "DevOps Specialist",
        "tech_stack": ["Azure", "Jenkins", "Bash"],
        "description": "Senior DevOps role focused on Azure cloud deployments and automation.",
        "posted_days_ago": 30,
    },
    "job_005": {
        "title": "Platform Engineer",
        "tech_stack": ["Kubernetes", "AWS", "ArgoCD", "TypeScript"],
        "description": "Build internal developer platforms and self-service infrastructure tooling.",
        "posted_days_ago": 3,
    },
    "job_006": {
        "title": "Junior Cloud Engineer",
        "tech_stack": ["AWS", "Linux", "Docker"],
        "description": "Entry-level role supporting AWS infrastructure. Great learning opportunity.",
        "posted_days_ago": 7,
    },
}

QUERY = "Senior DevOps Engineer Kubernetes AWS"


# =============================================================================
# STEP 2: Simulated Retrievers (Replace with real BM25/embeddings in production)
# =============================================================================
def simulate_sparse_search(query: str, jobs: dict, field: str) -> List[str]:
    """
    Simulates BM25-like keyword matching.
    In production: use Elasticsearch, Tantivy, or Qdrant sparse vectors.
    """
    query_terms = set(query.lower().split())
    scored = []
    for job_id, job in jobs.items():
        text = ""
        if field == "title":
            text = job["title"].lower()
        elif field == "desc":
            text = f"{job['description']} {' '.join(job['tech_stack'])}".lower()

        # Simple term overlap as proxy for BM25
        overlap = len(query_terms & set(text.split()))
        if overlap > 0:
            scored.append((job_id, overlap))

    # Sort by overlap descending (simulating BM25 ranking)
    scored.sort(key=lambda x: x[1], reverse=True)
    print(
        f"[LOG] Sparse ({field}) matched {len(scored)} docs: {[s[0] for s in scored]}"
    )
    return [doc_id for doc_id, _ in scored]


def simulate_dense_search(query: str, jobs: dict) -> List[str]:
    """
    Simulates semantic vector search.
    In production: use sentence-transformers + Qdrant/Weaviate dense index.

    NOTE: This intentionally returns semantically relevant but keyword-mismatched
    results to demonstrate why dense-only fails for job search.
    """
    # Hand-curated semantic rankings to illustrate the point
    # "Cloud Infrastructure Lead" and "SRE" are semantically close to "DevOps"
    # but lack exact keyword matches. "Junior Cloud Engineer" is semantically
    # related but wrong seniority - dense models often make this mistake.
    semantic_ranking = [
        "job_003",  # SRE - very close semantically to DevOps
        "job_002",  # Cloud Infra Lead - conceptually similar role
        "job_005",  # Platform Eng - modern synonym for DevOps
        "job_001",  # Exact match also scores well semantically
        "job_006",  # ⚠️ WRONG: Junior role, but "AWS cloud" is semantically close
        "job_004",  # Azure DevOps - related but different cloud provider
    ]
    print(f"[LOG] Dense (semantic) returned: {semantic_ranking}")
    return semantic_ranking


# =============================================================================
# STEP 3: Weighted RRF with Recency Decay (Job-Search Adapted)
# =============================================================================
def job_search_hybrid_rrf(
    title_results: List[str],
    desc_results: List[str],
    semantic_results: List[str],
    jobs: dict,
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    recency_half_life_days: float = 14.0,
) -> List[Tuple[str, float, Dict]]:
    """
    Job-search-adapted Hybrid RRF with:
    - Field-weighted fusion (title > description > semantic)
    - Recency decay to penalize stale postings
    - Detailed logging for transparency
    """
    if weights is None:
        weights = {"title": 1.5, "desc": 1.0, "semantic": 0.8}

    lists_config = [
        (title_results, weights["title"], "title"),
        (desc_results, weights["desc"], "desc"),
        (semantic_results, weights["semantic"], "semantic"),
    ]

    # Accumulate weighted RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)
    source_hits: Dict[str, List[str]] = defaultdict(list)

    for ranked_list, weight, source_name in lists_config:
        for rank, doc_id in enumerate(ranked_list):
            contribution = weight / (k + rank)
            rrf_scores[doc_id] += contribution
            source_hits[doc_id].append(f"{source_name}(rank={rank},w={weight})")
            print(
                f"  [RRF] {doc_id} += {contribution:.4f} from {source_name} rank {rank}"
            )

    # Apply recency decay: score * 2^(-days/half_life)
    import math

    final_results = []
    for doc_id, raw_score in rrf_scores.items():
        days_old = jobs[doc_id]["posted_days_ago"]
        decay = math.pow(2, -days_old / recency_half_life_days)
        adjusted_score = raw_score * decay
        final_results.append(
            (
                doc_id,
                adjusted_score,
                {
                    "raw_rrf": round(raw_score, 4),
                    "decay": round(decay, 4),
                    "days_old": days_old,
                    "sources": source_hits[doc_id],
                },
            )
        )
        print(
            f"  [DECAY] {doc_id}: {raw_score:.4f} * {decay:.4f} = {adjusted_score:.4f} ({days_old}d old)"
        )

    # Sort by adjusted score descending
    final_results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n[LOG] Final fused ranking: {len(final_results)} candidates")
    return final_results


import logging
import re
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class JobSparseIndex:
    """
    BM25 sparse index for job postings.
    Builds separate indexes for title and description+tech_stack fields
    to support field-weighted hybrid RRF.
    """

    def __init__(self):
        self.title_index: Optional[BM25Okapi] = None
        self.desc_index: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
        self._is_built = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return [t for t in cleaned.split() if len(t) > 1]

    def build(self, jobs: Dict[str, dict]) -> None:
        """
        Build BM25 indexes from job corpus.
        Call once at startup or after corpus updates.
        """
        self.doc_ids = list(jobs.keys())
        title_corpus = []
        desc_corpus = []

        for job_id in self.doc_ids:
            job = jobs[job_id]
            # Title tokens
            title_corpus.append(self._tokenize(job["title"]))
            # Description + tech stack tokens (combined for broader keyword match)
            desc_text = f"{job['description']} {' '.join(job.get('tech_stack', []))}"
            desc_corpus.append(self._tokenize(desc_text))

        self.title_index = BM25Okapi(title_corpus)
        self.desc_index = BM25Okapi(desc_corpus)
        self._is_built = True
        logger.info(f"[SPARSE] Built BM25 indexes for {len(self.doc_ids)} jobs")

    def search(self, query: str, field: str, top_k: int = 20) -> List[str]:
        """
        Search by field. Returns ranked list of job IDs.

        Args:
            query: Raw user query string
            field: "title" or "desc"
            top_k: Max results to return
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build() first.")

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            logger.warning(f"[SPARSE] Empty tokenized query: '{query}'")
            return []

        index = self.title_index if field == "title" else self.desc_index
        scores = index.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        # Filter out zero-score results
        results = [self.doc_ids[i] for i in top_indices if scores[i] > 0]

        logger.info(
            f"[SPARSE] field={field} query='{query}' → "
            f"{len(results)} hits (top score={scores[top_indices[0]]:.3f})"
        )
        return results


import logging
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model chosen specifically for short-text / job-description similarity.
# Alternatives: "BAAI/bge-small-en-v1.5" (faster), "intfloat/multilingual-e5-small" (multilingual)
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class JobDenseIndex:
    """
    Dense vector index for semantic job search.
    Encodes job postings once; computes cosine similarity at query time.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._is_built = False

    def build(self, jobs: Dict[str, dict], batch_size: int = 32) -> None:
        """
        Encode all job postings into dense vectors.
        Combines title + description + tech_stack for holistic representation.
        """
        logger.info(f"[DENSE] Loading model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)

        self.doc_ids = list(jobs.keys())
        texts = []
        for job_id in self.doc_ids:
            job = jobs[job_id]
            # Structured text format helps the model understand field boundaries
            text = (
                f"Job Title: {job['title']}. "
                f"Tech Stack: {', '.join(job.get('tech_stack', []))}. "
                f"Description: {job['description']}"
            )
            texts.append(text)

        logger.info(f"[DENSE] Encoding {len(texts)} job postings...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
        )
        self._is_built = True
        logger.info(
            f"[DENSE] Index built. Shape: {self.embeddings.shape}, "
            f"Dims: {self.embeddings.shape[1]}"
        )

    def search(self, query: str, top_k: int = 20) -> List[str]:
        """
        Semantic search via cosine similarity.

        Args:
            query: Raw user query string
            top_k: Max results to return
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build() first.")

        # Encode query (single sentence, no batching needed)
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        # Cosine similarity = dot product of L2-normalized vectors
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top_k indices sorted by similarity descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [self.doc_ids[i] for i in top_indices]
        top_score = similarities[top_indices[0]]

        logger.info(
            f"[DENSE] query='{query}' → {len(results)} results "
            f"(top sim={top_score:.4f})"
        )
        return results


# --- Build indexes once at startup ---
sparse_index = JobSparseIndex()
dense_index = JobDenseIndex()

sparse_index.build(JOB_POSTINGS)
dense_index.build(JOB_POSTINGS)


# --- Updated retrieval calls (same signature as simulate_*) ---
def real_sparse_search(query: str, jobs: dict, field: str) -> List[str]:
    """Drop-in replacement for simulate_sparse_search."""
    return sparse_index.search(query, field=field, top_k=20)


def real_dense_search(query: str, jobs: dict) -> List[str]:
    """Drop-in replacement for simulate_dense_search."""
    return dense_index.search(query, top_k=20)


# --- Use in hybrid RRF exactly as before ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print(f"QUERY: '{QUERY}'")
    print("=" * 70)

    title_hits = real_sparse_search(QUERY, JOB_POSTINGS, "title")
    desc_hits = real_sparse_search(QUERY, JOB_POSTINGS, "desc")
    dense_hits = real_dense_search(QUERY, JOB_POSTINGS)

    fused = job_search_hybrid_rrf(
        title_results=title_hits,
        desc_results=desc_hits,
        semantic_results=dense_hits,
        jobs=JOB_POSTINGS,
    )

    print("\n📌 FINAL RANKING:")
    for i, (doc_id, score, meta) in enumerate(fused, 1):
        job = JOB_POSTINGS[doc_id]
        print(
            f"  #{i} {doc_id} | {job['title']:<28} | score={score:.4f} | {meta['days_old']}d old"
        )
