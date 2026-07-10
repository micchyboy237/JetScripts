"""
10_demo_evaluation_ragas.py
Strategy: RAG evaluation (RAGAS-style metrics)

Problem it solves: "it looks right in the demo" isn't good enough for
production. Teams track four metrics to catch regressions: faithfulness
(is the answer actually grounded in retrieved context, or hallucinated?),
answer relevancy (does the answer address the question?), context
precision (how much of what was retrieved is actually relevant?), and
context recall (how much of the relevant material was retrieved at all?).

Use case: a pre-deployment eval harness / regression test for any RAG
pipeline, run against a small labeled set of (query, expected relevant
doc ids, generated answer) examples.

Note: RAGAS normally uses an LLM-as-judge for faithfulness/relevancy.
Here those are STUBs based on lexical term overlap — deterministic and
offline, but representative of what the LLM judge is actually checking.
"""
from common import TFIDFIndex, load_docs, tokenize

# A tiny labeled eval set: each item has a query, the doc ids that are
# actually relevant (ground truth), and a generated answer to score.
EVAL_SET = [
    {
        "query": "What is the water damage coverage limit?",
        "relevant_doc_ids": {"doc_003"},
        "generated_answer": "Water damage from burst pipes is covered up to $50,000 per incident.",
    },
    {
        "query": "How do I fail over the primary database?",
        "relevant_doc_ids": {"doc_005"},
        "generated_answer": "Promote the standby replica in us-east-2 after confirming the primary is unreachable.",
    },
]


def context_precision(retrieved_ids, relevant_ids):
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(retrieved_ids)


def context_recall(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 1.0
    hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return hits / len(relevant_ids)


def faithfulness_stub(answer, retrieved_docs):
    """STUB for LLM-judged faithfulness: what fraction of the answer's
    content words also appear somewhere in the retrieved context? Low
    overlap suggests the answer may contain ungrounded claims."""
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 1.0
    context_tokens = set()
    for d in retrieved_docs:
        context_tokens |= set(tokenize(d["text"]))
    grounded = sum(1 for t in answer_tokens if t in context_tokens)
    return grounded / len(answer_tokens)


def answer_relevancy_stub(answer, query):
    """STUB for LLM-judged relevancy: what fraction of the query's
    content words are addressed somewhere in the answer?"""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 1.0
    answer_tokens = set(tokenize(answer))
    addressed = sum(1 for t in query_tokens if t in answer_tokens)
    return addressed / len(query_tokens)


def main():
    docs = load_docs()
    index = TFIDFIndex(docs)

    header = f"{'query':45} {'faithfulness':>12} {'relevancy':>10} {'precision':>10} {'recall':>8}"
    print(header)
    print("-" * len(header))

    for case in EVAL_SET:
        results = index.search(case["query"], top_k=3)
        retrieved_docs = [d for d, _ in results]
        retrieved_ids = {d["id"] for d in retrieved_docs}

        faithfulness = faithfulness_stub(case["generated_answer"], retrieved_docs)
        relevancy = answer_relevancy_stub(case["generated_answer"], case["query"])
        precision = context_precision(retrieved_ids, case["relevant_doc_ids"])
        recall = context_recall(retrieved_ids, case["relevant_doc_ids"])

        print(f"{case['query'][:43]:45} {faithfulness:12.2f} {relevancy:10.2f} {precision:10.2f} {recall:8.2f}")

    print(
        "\nCommon production targets: faithfulness > 0.9, relevancy > 0.85, "
        "context precision > 0.8, context recall > 0.75."
    )


if __name__ == "__main__":
    main()
