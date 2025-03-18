import math
from collections import Counter
from typing import List, Dict, Optional, TypedDict
from nltk.stem import PorterStemmer
from thefuzz import fuzz

# Define SimilarityResult


class SimilarityResult(TypedDict):
    id: str
    text: str
    score: float
    matched: list[str]


class BM25SimilarityResult(SimilarityResult):
    similarity: Optional[float]


# Initialize the stemmer for partial word matching
stemmer = PorterStemmer()

# Function to perform fuzzy matching


def fuzzy_match(query_terms: List[str], doc_terms: List[str], threshold=80) -> List[str]:
    matched = []
    for query_term in query_terms:
        for doc_term in doc_terms:
            # Apply fuzzy match with threshold
            if fuzz.ratio(query_term, doc_term) >= threshold:
                matched.append(query_term)
                break  # Found a match, no need to check further
    return matched

# Modified get_bm25_similarities with heuristic reranker


def get_bm25_similarities(queries: List[str], documents: List[str], ids: List[str], *, k1=1.2, b=0.75, delta=1.0) -> List[BM25SimilarityResult]:
    # Tokenize documents
    tokenized_docs = [doc.split() for doc in documents]
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths)

    # Compute document frequency (DF)
    df = {}
    total_docs = len(documents)
    for doc in tokenized_docs:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1

    # Precompute IDF values
    idf = {term: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
           for term, freq in df.items()}

    all_scores: list[BM25SimilarityResult] = []

    for idx, doc in enumerate(tokenized_docs):
        doc_length = doc_lengths[idx]
        term_frequencies = Counter(doc)
        score = 0
        matched_queries = []

        for query in queries:
            query_terms = query.split()
            query_score = 0
            matched_terms = []

            # Exact word match
            for term in query_terms:
                if term in idf and term in doc:
                    matched_terms.append(term)
                    tf = term_frequencies[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * \
                        (1 - b + b * (doc_length / avg_doc_len)) + delta
                    query_score += idf[term] * (numerator / denominator)

            # Partial word match (stemming)
            if len(matched_terms) == 0:  # If no exact match, try stemming
                stemmed_query_terms = [stemmer.stem(
                    term) for term in query_terms]
                matched_terms = fuzzy_match(stemmed_query_terms, doc)
                if matched_terms:
                    query_score = 0  # Recalculate score based on partial match
                    for term in matched_terms:
                        tf = term_frequencies[term]
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * \
                            (1 - b + b * (doc_length / avg_doc_len)) + delta
                        query_score += idf[term] * (numerator / denominator)

            # Fuzzy match if still no result
            if len(matched_terms) == 0:  # If no exact or partial match, use fuzzy matching
                matched_terms = fuzzy_match(query_terms, doc)
                if matched_terms:
                    query_score = 0  # Recalculate score based on fuzzy match
                    for term in matched_terms:
                        tf = term_frequencies[term]
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * \
                            (1 - b + b * (doc_length / avg_doc_len)) + delta
                        query_score += idf[term] * (numerator / denominator)

            if query_score > 0:
                # Store the matched terms
                matched_queries.append(' '.join(matched_terms))

            score += query_score

        if score > 0:
            all_scores.append({
                "id": ids[idx],
                "score": score,
                "similarity": score,
                "matched": matched_queries,
                "text": documents[idx]
            })

    # Normalize scores based on the max score
    if all_scores:
        max_similarity = max(entry["score"] for entry in all_scores)
        for entry in all_scores:
            entry["score"] = entry["score"] / \
                max_similarity if max_similarity > 0 else 0

    # Sort results by normalized score in descending order
    return sorted(all_scores, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    queries = ["running fast", "goal oriented"]
    documents = ["I am running at a fast pace", "I have a goal in mind"]
    ids = ["1", "2"]

    result = get_bm25_similarities(queries, documents, ids)
    assert len(result) == 2
