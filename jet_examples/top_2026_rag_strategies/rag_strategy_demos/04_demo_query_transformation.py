"""
04_demo_query_transformation.py
Strategy: Query transformation — multi-query expansion + HyDE

Problem it solves: a user's raw query may not share vocabulary with
the documents that answer it. Expanding the query into several
phrasings (multi-query) or generating a hypothetical answer to embed
instead of the raw question (HyDE) increases the chance of matching
the right vocabulary.

Use case: consumer-facing chat assistants where user phrasing is
unpredictable ("pipe burst, will insurance pay?" vs. document
language "water damage caused by burst pipes is covered...").

Note: real systems generate paraphrases / hypothetical documents with
an LLM. These are STUBs — simple rule-based expansion/templating — so
the pipeline shape is demonstrated without any external API call.
"""
from common import TFIDFIndex, load_docs, print_results, reciprocal_rank_fusion

SYNONYMS = {
    "pipe": ["plumbing", "pipe"],
    "burst": ["burst", "broken", "leak"],
    "insurance": ["insurance", "policy", "coverage"],
    "pay": ["pay", "cover", "reimburse"],
    "claim": ["claim", "file a claim"],
    "deadline": ["deadline", "time limit", "days"],
}


def multi_query_expand_stub(query):
    """STUB for an LLM paraphrase step: naive synonym substitution to
    produce a few alternate phrasings of the same question."""
    variants = {query}
    words = query.lower().split()
    for i, w in enumerate(words):
        w_clean = w.strip("?.,")
        for syn in SYNONYMS.get(w_clean, []):
            new_words = words.copy()
            new_words[i] = syn
            variants.add(" ".join(new_words))
    return list(variants)


def hyde_stub(query):
    """STUB for an LLM HyDE step: generate a plausible hypothetical
    *answer* passage, then retrieve using that instead of the raw
    question, since answers use answer-like vocabulary that's closer
    to what's actually in the documents."""
    templates = {
        "claim": "Claims must be submitted within a set number of days of the incident, and late "
                 "claims may be denied unless there was a valid reason for the delay.",
        "pipe": "Water damage from a burst pipe is typically covered up to a policy limit, as long "
                "as the pipe was properly maintained.",
    }
    for key, hypothetical in templates.items():
        if key in query.lower():
            return hypothetical
    return query  # fall back to the raw query if no template matches


def main():
    docs = load_docs()
    index = TFIDFIndex(docs)
    query = "pipe burst, will insurance pay? what's the claim deadline"

    print(f"Original query: {query!r}\n")

    variants = multi_query_expand_stub(query)
    print(f"Multi-query expansion produced {len(variants)} variant(s):")
    for v in variants:
        print(f"  - {v}")

    variant_results = [index.search(v, top_k=5) for v in variants]
    fused = reciprocal_rank_fusion(variant_results, top_k=5)
    print_results("Fused results across all query variants", fused)

    hypothetical = hyde_stub(query)
    print(f"\nHyDE hypothetical passage: {hypothetical!r}")
    hyde_results = index.search(hypothetical, top_k=5)
    print_results("Retrieval using the HyDE hypothetical passage", hyde_results)


if __name__ == "__main__":
    main()
