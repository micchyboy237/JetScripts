"""
06_demo_multi_source_retrieval.py
Strategy: Multi-source / heterogeneous retrieval (breadth over depth)

Problem it solves: once a single retriever is decent, pushing top-k
deeper on it mostly returns redundant near-duplicates. What helps more
is combining retrievers/corpora that have *different failure modes* —
here, separate indexes per source type (policy docs, IT logs, product
catalog) — so a query surfaces relevant evidence regardless of which
corpus it lives in, instead of one large source dominating results.

Use case: an internal assistant whose knowledge is scattered across
formats — policy prose, structured logs, tabular catalog data — where
a single flat index tends to be dominated by whichever source has the
most documents.
"""
from collections import defaultdict

from common import TFIDFIndex, load_docs, print_results, reciprocal_rank_fusion


def build_source_indexes(docs):
    by_source = defaultdict(list)
    for d in docs:
        by_source[d.get("source", "unknown")].append(d)
    return {source: TFIDFIndex(source_docs) for source, source_docs in by_source.items()}


def main():
    docs = load_docs()
    indexes = build_source_indexes(docs)
    print(f"Built {len(indexes)} source-specific indexes: {list(indexes.keys())}")

    query = "system incident and recovery steps"

    per_source_results = []
    for source, index in indexes.items():
        results = index.search(query, top_k=3)
        if results:
            print_results(f"Source: {source}", results)
            per_source_results.append(results)

    fused = reciprocal_rank_fusion(per_source_results, top_k=5)
    print_results("Fused across all sources (RRF)", fused)


if __name__ == "__main__":
    main()
