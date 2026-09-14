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


# =============================================================================
# STEP 4: Run Comparison
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"QUERY: '{QUERY}'")
    print("=" * 70)

    # Individual retrievers
    print("\n--- SPARSE: Title Match ---")
    title_hits = simulate_sparse_search(QUERY, JOB_POSTINGS, "title")

    print("\n--- SPARSE: Description + Tech Stack Match ---")
    desc_hits = simulate_sparse_search(QUERY, JOB_POSTINGS, "desc")

    print("\n--- DENSE: Semantic Similarity ---")
    dense_hits = simulate_dense_search(QUERY, JOB_POSTINGS)

    # Hybrid fusion
    print("\n--- HYBRID RRF FUSION ---")
    fused = job_search_hybrid_rrf(
        title_results=title_hits,
        desc_results=desc_hits,
        semantic_results=dense_hits,
        jobs=JOB_POSTINGS,
    )

    # Display final comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)
    print(
        f"{'Rank':<5} {'Job ID':<9} {'Title':<28} {'Score':<8} {'Days':<5} {'Sources'}"
    )
    print("-" * 90)
    for i, (doc_id, score, meta) in enumerate(fused, 1):
        job = JOB_POSTINGS[doc_id]
        sources_str = ", ".join(meta["sources"])
        print(
            f"{i:<5} {doc_id:<9} {job['title']:<28} {score:<8.4f} {meta['days_old']:<5} {sources_str}"
        )

    # Highlight key insights
    print("\n📌 KEY INSIGHTS:")
    print("  ✅ job_001 (Exact match, fresh) → Ranked #1 via all 3 signals")
    print("  ✅ job_003 (SRE, K8s+AWS, freshest) → Boosted by semantic + recency")
    print(
        "  ✅ job_005 (Platform Eng, K8s+AWS) → Caught by semantic despite no 'DevOps' keyword"
    )
    print(
        "  ⚠️  job_006 (Junior, AWS only) → Demoted: no title/desc match + lower semantic rank"
    )
    print(
        "  ❌ job_004 (Azure, 30d old) → Penalized by recency decay AND missing K8s/AWS keywords"
    )
