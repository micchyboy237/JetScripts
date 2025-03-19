import math
from collections import Counter
from typing import List, Dict, Optional, TypedDict
from jet.file.utils import load_file
from jet.logger import logger, time_it
from jet.search.transformers import clean_string
from jet.utils.commands import copy_to_clipboard
from jet.utils.object import extract_values_by_paths
from jet.wordnet.n_grams import get_most_common_ngrams
from nltk.stem import PorterStemmer
from shared.data_types.job import JobData
from jet.cache.cache_manager import CacheManager
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
@time_it
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

    # Compute IDF values with fallback for missing terms
    idf = {term: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
           for term, freq in df.items()}
    default_idf = math.log((total_docs + 0.5) / 0.5 + 1)  # Fallback IDF

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

            # Filter query terms to avoid KeyError
            query_terms = [term for term in query_terms if term in idf]

            # Exact word match
            for term in query_terms:
                # Use default IDF if missing
                term_idf = idf.get(term, default_idf)
                if term in doc:
                    matched_terms.append(term)
                    tf = term_frequencies[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * \
                        (1 - b + b * (doc_length / avg_doc_len)) + delta
                    query_score += term_idf * (numerator / denominator)

            # Partial word match (stemming)
            if not matched_terms:  # If no exact match, try stemming
                stemmed_query_terms = [stemmer.stem(
                    term) for term in query_terms]
                matched_terms = fuzzy_match(stemmed_query_terms, doc)
                if matched_terms:
                    query_score = 0  # Recalculate score based on partial match
                    for term in matched_terms:
                        term_idf = idf.get(term, default_idf)
                        tf = term_frequencies[term]
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * \
                            (1 - b + b * (doc_length / avg_doc_len)) + delta
                        query_score += term_idf * (numerator / denominator)

            # Fuzzy match if still no result
            if not matched_terms:  # If no exact or partial match, use fuzzy matching
                matched_terms = fuzzy_match(query_terms, doc)
                if matched_terms:
                    query_score = 0  # Recalculate score based on fuzzy match
                    for term in matched_terms:
                        term_idf = idf.get(term, default_idf)
                        tf = term_frequencies[term]
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * \
                            (1 - b + b * (doc_length / avg_doc_len)) + delta
                        query_score += term_idf * (numerator / denominator)

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


@time_it
def prepare_inputs(queries: list[str]):
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    cache_dir = "generated/get_bm25_similarities"
    data: list[JobData] = load_file(data_file)

    cache_manager = CacheManager(cache_dir=cache_dir)

    # Load previous cache data
    cache_data = cache_manager.load_cache()

    if not cache_manager.is_cache_valid(data_file, cache_data):
        sentences = []
        for item in data:
            sentence = "\n".join([
                item["title"],
                item["details"],
                "\n".join([f"Tech: {tech}" for tech in sorted(
                    item["entities"]["technology_stack"], key=str.lower)]),
                "\n".join([f"Tag: {tech}" for tech in sorted(
                    item["tags"], key=str.lower)]),
            ])
            cleaned_sentence = clean_string(sentence.lower())
            sentences.append(cleaned_sentence)

        # Generate n-grams
        common_texts_ngrams = [
            list(get_most_common_ngrams(sentence, max_words=5).keys()) for sentence in sentences
        ]
    else:
        # Use the cached n-grams
        common_texts_ngrams = cache_data["common_texts_ngrams"]

    # Prepare queries and calculate BM25+ similarities
    query_ngrams = [list(get_most_common_ngrams(
        query, min_count=1, max_words=5)) for query in queries]
    data_dict = {item["id"]: item for item in data}
    ids = list(data_dict.keys())
    queries = ["_".join(text.split())
               for queries in query_ngrams for text in queries]

    common_texts = []
    for texts in common_texts_ngrams:
        formatted_texts = []
        for text in texts:
            formatted_texts.append("_".join(text.split()))
        common_texts.append(" ".join(formatted_texts))

    return data, {
        "queries": queries,
        "documents": common_texts,
        "ids": ids,
    }


if __name__ == "__main__":
    queries = [
        "React.js",
        "React Native",
        "Web",
        "Mobile"
    ]

    data, inputs_dict = prepare_inputs(queries)
    data_dict = {item["id"]: item for item in data}

    similarities = get_bm25_similarities(**inputs_dict)
    # Format the results
    results = [
        {
            "score": result["score"],
            "similarity": result["similarity"],
            "matched": result["matched"],
            "result": data_dict[result["id"]]
        }
        for result in similarities
    ]

    copy_to_clipboard({
        "count": len(results),
        "data": results
    })

    for idx, data in enumerate(results[:10]):
        result = data["result"]
        logger.log(f"{idx + 1}:", result["title"]
                   [:30], colors=["WHITE", "DEBUG"])
        logger.success(f"{result["score"]:.2f}")
