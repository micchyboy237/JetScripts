from rank_bm25 import BM25Okapi
from jet.vectors.reranker.bm25_okapi import prepare_corpus, get_top_k_results, inspect_idf


def example_init():
    tokenized_corpus, raw_corpus = prepare_corpus()

    # Initialize with default parameters
    bm25_default = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75, epsilon=0.25)
    print("Default BM25Okapi initialized with k1=1.5, b=0.75, epsilon=0.25")

    # Initialize with custom epsilon for frequent terms
    bm25_custom = BM25Okapi(tokenized_corpus, k1=1.2, b=0.5, epsilon=0.5)
    print("Custom BM25Okapi initialized with k1=1.2, b=0.5, epsilon=0.5")

    # Verify initialization by checking average_idf
    print(f"Default model average IDF: {bm25_default.average_idf:.4f}")
    print(f"Custom model average IDF: {bm25_custom.average_idf:.4f}")


def example_calc_idf():
    tokenized_corpus, _ = prepare_corpus()

    # Initialize with different epsilon values
    bm25_low_eps = BM25Okapi(tokenized_corpus, epsilon=0.1)
    bm25_high_eps = BM25Okapi(tokenized_corpus, epsilon=0.5)

    # Check IDF for a frequent term (e.g., "is" appears in multiple documents)
    terms = ["is", "natural", "language"]
    idf_low_eps = inspect_idf(bm25_low_eps, terms)
    idf_high_eps = inspect_idf(bm25_high_eps, terms)

    print("IDF values with epsilon=0.1:")
    for term, idf in idf_low_eps.items():
        print(f"Term: {term}, IDF: {idf:.4f}")

    print("\nIDF values with epsilon=0.5:")
    for term, idf in idf_high_eps.items():
        print(f"Term: {term}, IDF: {idf:.4f}")

    print(f"\nAverage IDF (epsilon=0.1): {bm25_low_eps.average_idf:.4f}")
    print(f"Average IDF (epsilon=0.5): {bm25_high_eps.average_idf:.4f}")


def example_get_scores():
    tokenized_corpus, raw_corpus = prepare_corpus()
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75, epsilon=0.25)

    # Define a multi-term query
    query = "natural language processing text mining".lower().split()

    # Get scores for all documents
    scores = bm25.get_scores(query)

    print("BM25Okapi Scores for all documents:")
    for idx, (score, doc) in enumerate(zip(scores, raw_corpus)):
        print(f"Index: {idx}, Score: {score:.4f}")
        print(f"Document: {doc}\n")


def example_get_batch_scores():
    tokenized_corpus, raw_corpus = prepare_corpus()
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75, epsilon=0.25)

    # Define a multi-term query
    query = "natural language processing text mining".lower().split()

    # Select a subset of document indices
    doc_ids = [1, 3, 4]  # Only score documents 1, 3, and 4

    # Get batch scores
    scores = bm25.get_batch_scores(query, doc_ids)

    print("BM25Okapi Batch Scores for selected documents:")
    for idx, score in zip(doc_ids, scores):
        print(f"Index: {idx}, Score: {score:.4f}")
        print(f"Document: {raw_corpus[idx]}\n")


def example_combined():
    tokenized_corpus, raw_corpus = prepare_corpus()

    # Initialize with different epsilon values to see effect
    bm25_low_eps = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75, epsilon=0.1)
    bm25_high_eps = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75, epsilon=0.5)

    # Include frequent term "is"
    query = "natural language processing text mining is".lower().split()

    # Get top-3 results
    print("=== BM25Okapi (epsilon=0.1) Results ===")
    for result in get_top_k_results(bm25_low_eps, query, raw_corpus, k=3):
        print(f"Index: {result['index']}, Score: {result['score']:.4f}")
        print(f"Document: {result['document']}\n")

    print("=== BM25Okapi (epsilon=0.5) Results ===")
    for result in get_top_k_results(bm25_high_eps, query, raw_corpus, k=3):
        print(f"Index: {result['index']}, Score: {result['score']:.4f}")
        print(f"Document: {result['document']}\n")

    # Inspect IDFs to see epsilon's effect
    terms = ["is", "natural", "language"]
    print("IDF values (epsilon=0.1):", inspect_idf(bm25_low_eps, terms))
    print("IDF values (epsilon=0.5):", inspect_idf(bm25_high_eps, terms))


if __name__ == "__main__":
    example_init()
    example_calc_idf()
    example_get_scores()
    example_get_batch_scores()
    example_combined()
