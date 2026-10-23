from collections import defaultdict

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# 1. Sample Data
# -------------------------------------------------
documents = [
    "The cat sat on the mat in the living room",
    "A dog played with a ball in the park",
    "Cats and dogs are popular household pets",
    "I love eating fresh red apples every morning",
    "The feline rested comfortably on the soft rug",
    "Python is a great programming language for AI",
]

doc_ids = [f"doc_{i}" for i in range(len(documents))]

query = "cat sitting on a rug"

print("Query:", query)
print("-" * 60)

# -------------------------------------------------
# 2. Sparse Retrieval (BM25)
# -------------------------------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

tokenized_query = query.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)

# Get top results from sparse
sparse_ranked = sorted(zip(doc_ids, bm25_scores), key=lambda x: x[1], reverse=True)

print("\n Sparse (BM25) Results:")
for doc_id, score in sparse_ranked:
    print(f"  {doc_id}: {score:.4f} → {documents[int(doc_id.split('_')[1])]}")

# -------------------------------------------------
# 3. Dense Retrieval (Embeddings)
# -------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

dense_scores = cosine_similarity(query_embedding, doc_embeddings).flatten()

dense_ranked = sorted(zip(doc_ids, dense_scores), key=lambda x: x[1], reverse=True)

print("\n Dense Results:")
for doc_id, score in dense_ranked:
    print(f"  {doc_id}: {score:.4f} → {documents[int(doc_id.split('_')[1])]}")


# -------------------------------------------------
# 4. Reciprocal Rank Fusion (RRF)
# -------------------------------------------------
def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    ranked_lists: list of lists of doc_ids (already sorted by relevance)
    """
    scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    # Sort by fused score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Prepare ranked lists of doc_ids only
sparse_doc_ids = [doc_id for doc_id, _ in sparse_ranked]
dense_doc_ids = [doc_id for doc_id, _ in dense_ranked]

rrf_results = reciprocal_rank_fusion([sparse_doc_ids, dense_doc_ids], k=60)

print("\n RRF (Reciprocal Rank Fusion) Results:")
for doc_id, score in rrf_results:
    print(f"  {doc_id}: {score:.4f} → {documents[int(doc_id.split('_')[1])]}")


# -------------------------------------------------
# 5. Weighted Score Fusion
# -------------------------------------------------
def weighted_fusion(sparse_scores_dict, dense_scores_dict, alpha=0.4):
    """
    alpha = weight for sparse score
    (1 - alpha) = weight for dense score
    """
    all_docs = set(sparse_scores_dict.keys()) | set(dense_scores_dict.keys())
    fused = {}

    for doc_id in all_docs:
        s_score = sparse_scores_dict.get(doc_id, 0)
        d_score = dense_scores_dict.get(doc_id, 0)
        fused[doc_id] = alpha * s_score + (1 - alpha) * d_score

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# Convert to dictionaries for easy lookup
sparse_dict = dict(sparse_ranked)
dense_dict = dict(dense_ranked)

# Note: BM25 scores and cosine scores are on different scales.
# In real systems you should normalize them first (min-max or z-score).
weighted_results = weighted_fusion(sparse_dict, dense_dict, alpha=0.4)

print("\n Weighted Fusion Results (alpha=0.4):")
for doc_id, score in weighted_results:
    print(f"  {doc_id}: {score:.4f} → {documents[int(doc_id.split('_')[1])]}")
