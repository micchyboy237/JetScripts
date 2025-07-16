from jet.vectors.semantic_search.vector_search_simple import VectorSearch


if __name__ == "__main__":
    # Real-world demonstration
    search_engine = VectorSearch()

    # Same sample documents
    sample_docs = [
        "Fresh organic apples from local farms",
        "Handpicked strawberries sweet and juicy",
        "Premium quality oranges rich in vitamin C",
        "Crisp lettuce perfect for salads",
        "Organic bananas ripe and ready to eat"
    ]

    search_engine.add_documents(sample_docs)

    # Same example queries
    queries = [
        "organic fruit",
        "sweet strawberries",
        "fresh salad ingredients"
    ]

    for query in queries:
        results = search_engine.search(query)
        print(f"\nQuery: {query}")
        print("Top matches:")
        for doc, score in results:
            print(f"- {doc} (score: {score:.3f})")
