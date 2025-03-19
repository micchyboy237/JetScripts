import math
from collections import Counter
from typing import List, Dict


def get_bm25p_with_auto_penalty(queries: List[str], documents: List[str], ids: List[str], *,
                                penalty_factor=0.5, k1=1.2, b=0.75, delta=1.0) -> List[Dict]:
    """
    Compute BM25+ similarities between queries and a list of job applications.
    Automatically penalizes terms that appear in lower-ranked documents.

    Args:
        queries (List[str]): List of query strings.
        documents (List[str]): List of document strings.
        ids (List[str]): List of document ids corresponding to the documents.
        penalty_factor (float): Score reduction multiplier for automatically detected penalty terms.
        k1 (float): Term frequency scaling factor.
        b (float): Length normalization parameter.
        delta (float): BM25+ correction factor.

    Returns:
        List[Dict]: Sorted list of job applications ranked by relevance.
    """

    # Tokenize documents
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

    scores = []

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

        scores.append({"id": ids[idx], "score": score,
                      "text": documents[idx], "tokens": doc})

    # Step 1: Get terms that appear frequently in LOW-SCORING documents
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    top_docs = sorted_scores[:max(1, len(sorted_scores) // 3)]  # Top 1/3 docs
    # Bottom 1/3 docs
    bottom_docs = sorted_scores[-max(1, len(sorted_scores) // 3):]

    top_terms = Counter([term for doc in top_docs for term in doc["tokens"]])
    bottom_terms = Counter(
        [term for doc in bottom_docs for term in doc["tokens"]])

    # Find terms that appear significantly more in bottom-ranked docs
    penalty_terms = {
        term for term, count in bottom_terms.items() if count > top_terms.get(term, 0) * 2
    }

    # Step 2: Apply penalty to docs containing these terms
    for entry in scores:
        if penalty_terms.intersection(entry["tokens"]):
            # Reduce score if penalty terms are present
            entry["score"] *= penalty_factor

    # Normalize scores
    max_score = max(entry["score"] for entry in scores) if scores else 1
    for entry in scores:
        entry["score"] = entry["score"] / max_score if max_score > 0 else 0

    # Return re-ranked results
    return sorted(scores, key=lambda x: x["score"], reverse=True)


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

    # Run auto-penalizing BM25+
    results = get_bm25p_with_auto_penalty(
        queries, documents, ids, penalty_factor=0.5)

    for result in results:
        print(
            f"ID: {result['id']}, Score: {result['score']:.2f}, Text: {result['text']}")
