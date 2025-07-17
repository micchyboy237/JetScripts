from jet.vectors.semantic_search.vector_search_simple import VectorSearch


if __name__ == "__main__":
    # Real-world demonstration
    search_engine = VectorSearch("nomic-embed-text")

    # Same sample documents
    sample_docs = [
        "React.js",
        "Web",
        "Website",
        "Front end",
        "Back end",
        "Client side",
        "Server",
    ]

    search_engine.add_documents(sample_docs)

    # Same example queries
    queries = [
        "React web",
    ]

    for query in queries:
        results = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for doc, score in results:
            print(f"- {doc} (score: {score:.3f})")
