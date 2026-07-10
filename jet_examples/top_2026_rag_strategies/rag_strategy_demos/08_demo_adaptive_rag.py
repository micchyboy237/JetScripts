"""
08_demo_adaptive_rag.py
Strategy: Adaptive RAG (complexity-based query routing)

Problem it solves: running the full expensive pipeline (query
transformation, agentic decomposition, reranking) on every query is
wasteful — most queries are simple lookups. A lightweight classifier
routes each query to the cheapest pipeline that can actually answer it.

Use case: a support/internal assistant fielding a mix of trivial
lookups ("what's the SKU-4471 price?") and genuinely complex multi-hop
questions — you want to keep the simple case cheap and fast while
still handling the hard case well.

Note: a real system might train a small classifier or use an LLM call.
Here it's a STUB: a rule-based heuristic on query features (length,
conjunctions, comparison words) — no external API call.
"""
from common import TFIDFIndex, load_docs, print_results

COMPLEX_SIGNALS = ["and", "compare", "versus", " vs ", "relate", "why", "how does"]


def classify_complexity_stub(query):
    """STUB for a query-complexity classifier."""
    q = query.lower()
    word_count = len(q.split())
    signal_hits = sum(1 for s in COMPLEX_SIGNALS if s in q)
    if word_count <= 8 and signal_hits == 0:
        return "simple"
    if signal_hits >= 1 or word_count > 15:
        return "complex"
    return "moderate"


def simple_pipeline(index, query):
    """Single retrieval call, top-1 result — cheapest path."""
    return index.search(query, top_k=1)


def moderate_pipeline(index, query):
    """Single retrieval call, wider top-k for a bit more coverage."""
    return index.search(query, top_k=3)


def complex_pipeline(index, query):
    """Would hand off to the full agentic/decomposition pipeline (see
    07_demo_agentic_rag.py) in a complete system; here we widen
    retrieval further to represent the heavier path."""
    return index.search(query, top_k=5)


ROUTES = {"simple": simple_pipeline, "moderate": moderate_pipeline, "complex": complex_pipeline}


def main():
    docs = load_docs()
    index = TFIDFIndex(docs)

    queries = [
        "SKU-4471 price",
        "What is the water damage coverage limit?",
        "Can ACME Corp terminate the CloudStore contract and how does that relate to its revenue trend?",
    ]

    for query in queries:
        complexity = classify_complexity_stub(query)
        pipeline = ROUTES[complexity]
        print(f"\nQuery: {query!r}\nRouted to: {complexity.upper()} pipeline ({pipeline.__name__})")
        results = pipeline(index, query)
        print_results("Results", results)


if __name__ == "__main__":
    main()
