from jet.features.nltk_search import search_by_pos

# Example usage
if __name__ == "__main__":
    # Sample documents
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A fox fled from danger",
        "The dog sleeps peacefully",
        "Quick foxes climb steep hills"
    ]

    # Sample query (valid sentence)
    query = "The quick foxes run dangerously"

    # Get results
    results = search_by_pos(query, docs)

    # Print results
    for doc_idx, match_count, matches_with_pos in results:
        print(f"Document {doc_idx}:")
        print(f"  Text: {docs[doc_idx]}")
        print(f"  Matching words (word, POS, lemma): {matches_with_pos}")
        print(f"  Match count: {match_count}\n")
