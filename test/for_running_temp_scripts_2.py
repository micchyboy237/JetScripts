import math
from collections import Counter
from typing import List, TypedDict
from nltk.tokenize import word_tokenize


class BM25SimilarityResult(TypedDict):
    id: str
    score: float
    similarity: float
    matched: List[str]
    text: str


def get_bm25_similarities(queries: List[str], documents: List[str], ids: List[str], *, k1=1.2, b=0.75, delta=1.0) -> List[BM25SimilarityResult]:
    """
    Compute BM25+ similarities between queries and a list of documents.

    Args:
        queries (List[str]): List of query strings.
        documents (List[str]): List of document strings.
        ids (List[str]): List of document ids corresponding to the documents.
        k1 (float): Term frequency scaling factor.
        b (float): Length normalization parameter.
        delta (float): BM25+ correction factor to reduce the bias against short documents.

    Returns:
        List[BM25SimilarityResult]: A list of dictionaries containing scores, similarities, matched queries, ids, and text.
    """
    # Tokenize queries & documents
    tokenized_queries = [word_tokenize(q.lower()) for q in queries]
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths)

    # Compute Document Frequency (DF)
    df = {}
    total_docs = len(documents)

    for doc in tokenized_docs:
        for term in set(doc):  # Count only unique terms per document
            df[term] = df.get(term, 0) + 1

    # Precompute IDF values
    idf = {term: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
           for term, freq in df.items()}

    all_scores: List[BM25SimilarityResult] = []

    for idx, doc in enumerate(tokenized_docs):
        doc_length = doc_lengths[idx]
        term_frequencies = Counter(doc)
        score = 0
        matched_queries = []

        for query in tokenized_queries:
            query_score = 0

            for term in query:
                if term in idf:
                    tf = term_frequencies[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * \
                        (1 - b + b * (doc_length / avg_doc_len)) + delta
                    query_score += idf[term] * (numerator / denominator)

            if query_score > 0:
                matched_queries.append(" ".join(query))

            score += query_score

        if score > 0:
            all_scores.append({
                "id": ids[idx],
                "score": score,
                "similarity": score,
                "matched": matched_queries,
                "text": documents[idx]
            })

    # Normalize scores
    if all_scores:
        max_similarity = max(entry["score"] for entry in all_scores)
        for entry in all_scores:
            entry["score"] /= max_similarity if max_similarity > 0 else 1

    return sorted(all_scores, key=lambda x: x["score"], reverse=True)
