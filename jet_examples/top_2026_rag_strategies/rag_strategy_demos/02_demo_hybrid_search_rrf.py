"""
02_demo_hybrid_search_rrf.py
Strategy: Hybrid search (dense TF-IDF + sparse BM25) fused with
Reciprocal Rank Fusion (RRF)

Problem it solves: pure dense/semantic retrieval misses exact-match
queries (SKUs, names, policy numbers); pure sparse (BM25) retrieval
misses paraphrases. Running both in parallel and fusing by *rank*
(RRF) rather than raw score (dense and BM25 scores aren't on
comparable scales) captures the benefit of each.

Use case: enterprise / product search where queries mix free text
("water bottle warranty") with exact identifiers ("SKU-4471").
"""
from common import BM25Index, TFIDFIndex, load_docs, print_results, reciprocal_rank_fusion


def main():
    docs = load_docs()
    dense_index = TFIDFIndex(docs)
    sparse_index = BM25Index(docs)

    query = "SKU-4471 warranty and stock"
    print(f"Query: {query!r}")

    dense_results = dense_index.search(query, top_k=5)
    print_results("Dense (TF-IDF cosine) only", dense_results)

    sparse_results = sparse_index.search(query, top_k=5)
    print_results("Sparse (BM25) only", sparse_results)

    fused = reciprocal_rank_fusion([dense_results, sparse_results], top_k=5)
    print_results("Hybrid (RRF fusion of dense + sparse)", fused)


if __name__ == "__main__":
    main()
