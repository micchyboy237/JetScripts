import math
from collections import Counter
from typing import List, Dict


def get_bm25p_similarities(queries: List[str], documents: List[str], ids: List[str], *,
                           penalty_terms: List[str] = None, penalty_factor=0.5,
                           k1=1.2, b=0.75, delta=1.0) -> List[Dict]:
    """
    Compute BM25+ similarities between queries and a list of job applications.
    Dynamically penalizes documents containing user-defined terms.

    Args:
        queries (List[str]): List of query strings.
        documents (List[str]): List of document strings.
        ids (List[str]): List of document ids corresponding to the documents.
        penalty_terms (List[str]): List of terms to penalize (e.g., ["React Native"]).
        penalty_factor (float): Score reduction multiplier if penalty terms are present.
        k1 (float): Term frequency scaling factor.
        b (float): Length normalization parameter.
        delta (float): BM25+ correction factor to reduce short doc bias.

    Returns:
        List[Dict]: Sorted list of job applications ranked by relevance.
    """

    penalty_terms = set(term.lower()
                        for term in (penalty_terms or []))  # Normalize terms
    tokenized_docs = [doc.lower().split() for doc in documents]
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths)

    df = {}
    total_docs = len(documents)

    for doc in tokenized_docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    idf = {term: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
           for term, freq in df.items()}

    results = []

    for idx, doc in enumerate(tokenized_docs):
        doc_length = doc_lengths[idx]
        term_frequencies = Counter(doc)
        score = 0
        matched_queries = []

        for query in queries:
            query_terms = query.lower().split()
            query_score = 0

            for term in query_terms:
                if term in idf:
                    tf = term_frequencies[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * \
                        (1 - b + b * (doc_length / avg_doc_len)) + delta
                    query_score += idf[term] * (numerator / denominator)

            if query_score > 0:
                matched_queries.append(query)

            score += query_score

        # Apply penalty if any penalty term is found in the document
        if penalty_terms.intersection(doc):
            score *= penalty_factor  # Reduce score by penalty factor

        if score > 0:
            results.append({
                "id": ids[idx],
                "score": score,
                "matched": matched_queries,
                "text": documents[idx]
            })

    # Normalize scores
    if results:
        max_score = max(entry["score"] for entry in results)
        for entry in results:
            entry["score"] = entry["score"] / max_score if max_score > 0 else 0

    return sorted(results, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    queries = ["React web"]
    documents = [
        "React web developer needed for e-commerce site.",
        "React Native engineer required for mobile app development.",
        "Frontend engineer with strong React web skills needed.",
        "React and Redux developer required for dashboard project.",
        "React Native and React web hybrid developer position available."
    ]
    ids = ["job1", "job2", "job3", "job4", "job5"]

    # User-specified penalty terms
    penalty_terms = ["React Native"]

    # Run BM25+ with dynamic penalties
    results = get_bm25p_similarities(
        queries, documents, ids, penalty_terms=penalty_terms, penalty_factor=0.5)

    for result in results:
        print(
            f"ID: {result['id']}, Score: {result['score']:.2f}, Text: {result['text']}")
