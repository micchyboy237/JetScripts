"""
03_demo_reranking.py
Strategy: Cross-encoder reranking on top of first-stage retrieval

Problem it solves: first-stage retrievers (dense/BM25/hybrid) score
each document independently, which is fast but coarse. A reranker
jointly scores each (query, doc) pair for higher precision, at higher
per-pair cost — so it only runs on a small top-k shortlist, not the
whole corpus.

Use case: support/knowledge-base search, where the first-stage
retriever returns a decent shortlist but the *order* of the top 3-5
results is what the user actually sees.

Note: a real system would use a trained cross-encoder model (a
sentence-pair classifier). Here we use a transparent heuristic scorer
— weighted query-term coverage + term proximity — to demonstrate the
pipeline shape without any external model or API call.
"""
from common import TFIDFIndex, load_docs, print_results, tokenize


def cross_encoder_stub(query, doc_text):
    """STUB for a real cross-encoder: scores a (query, doc) pair
    jointly, rewarding coverage of query terms AND how close together
    those matches occur in the document, not just aggregate overlap."""
    q_tokens = set(tokenize(query))
    d_tokens = tokenize(doc_text)
    if not q_tokens or not d_tokens:
        return 0.0

    d_set = set(d_tokens)
    coverage = sum(1 for t in q_tokens if t in d_set) / len(q_tokens)

    positions = [i for i, t in enumerate(d_tokens) if t in q_tokens]
    if len(positions) > 1:
        proximity = 1.0 / (1 + (max(positions) - min(positions)))
    elif positions:
        proximity = 1.0
    else:
        proximity = 0.0

    return round(0.7 * coverage + 0.3 * proximity, 4)


def rerank(query, candidates, top_k=5):
    rescored = [(doc, cross_encoder_stub(query, doc["text"])) for doc, _first_stage_score in candidates]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_k]


def main():
    docs = load_docs()
    index = TFIDFIndex(docs)
    query = "how long do I have to file a water damage claim"

    first_stage = index.search(query, top_k=6)
    print_results("First-stage retrieval (TF-IDF, coarse)", first_stage)

    reranked = rerank(query, first_stage, top_k=3)
    print_results("After cross-encoder reranking (top 3)", reranked)


if __name__ == "__main__":
    main()
