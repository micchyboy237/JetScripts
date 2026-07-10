"""
07_demo_agentic_rag.py
Strategy: Agentic RAG (query decomposition + iterative retrieve-evaluate loop)

Problem it solves: some questions can't be answered by a single
retrieval call because the evidence for each part of the question
lives in different documents. An agentic loop breaks the question into
sub-questions, retrieves for each, checks whether it has enough
evidence, and stops once each sub-question is covered (or a max
iteration count is hit).

Use case: multi-hop due-diligence / compliance questions — e.g. "Can
ACME Corp terminate its CloudStore contract, and how would that
interact with its recent revenue trend?" — where no single chunk
answers the whole thing.

Note: a real system uses an LLM to decompose the query and judge
sufficiency. Here those two steps are STUBs (simple heuristics) so the
control-flow shape is demonstrated without any external API call.
"""
from common import TFIDFIndex, load_docs, print_results

MAX_ATTEMPTS_PER_SUBQUERY = 2


def decompose_query_stub(query):
    """STUB for an LLM decomposition step: split a compound question on
    conjunctions into independently-retrievable sub-questions."""
    for splitter in [", and", " and how", " and "]:
        if splitter in query:
            parts = query.split(splitter)
            return [p.strip().rstrip("?") + "?" for p in parts if p.strip()]
    return [query]


def sufficiency_check_stub(results, min_score=0.05):
    """STUB for an LLM 'do I have enough evidence' judgment: treat the
    top retrieved score as a proxy for whether the sub-question was
    actually answered by what was retrieved."""
    return bool(results) and results[0][1] >= min_score


def agentic_retrieve(index, query):
    sub_queries = decompose_query_stub(query)
    print(f"Decomposed into {len(sub_queries)} sub-question(s): {sub_queries}")

    evidence = {}
    for sub_q in sub_queries:
        results = []
        for attempt in range(1, MAX_ATTEMPTS_PER_SUBQUERY + 1):
            results = index.search(sub_q, top_k=3)
            print_results(f"Sub-question: {sub_q!r} (attempt {attempt})", results)
            if sufficiency_check_stub(results):
                break
        evidence[sub_q] = results
    return evidence


def main():
    docs = load_docs()
    index = TFIDFIndex(docs)

    query = (
        "Can ACME Corp terminate its CloudStore contract, and how does that relate "
        "to its recent revenue trend?"
    )
    print(f"Complex query: {query!r}\n")

    evidence = agentic_retrieve(index, query)

    print("\n=== Final assembled evidence set ===")
    for sub_q, results in evidence.items():
        top_doc = results[0][0]["id"] if results else "none"
        print(f"  {sub_q!r} -> best evidence: {top_doc}")


if __name__ == "__main__":
    main()
