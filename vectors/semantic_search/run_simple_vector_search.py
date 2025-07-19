from jet.vectors.semantic_search.vector_search_simple import VectorSearch


if __name__ == "__main__":
    # Real-world demonstration
    search_engine = VectorSearch("mxbai-embed-large")

    # Same sample documents
    sample_docs = [
        "## Keywords\n- React.js\n- AWS\n- React Native\n- Node.js",
        "## Technology Stack\n- React Native\n- React.js",
        "## Keywords\n- React.js",
        "## Technology Stack\n- React.js",
    ]

    search_engine.add_documents(sample_docs)

    # Same example queries
    queries = [
        "React Native",
    ]

    for query in queries:
        results = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for num, (doc, score) in enumerate(results, 1):
            print(f"\n{num}. (Score: {score:.3f})")
            print(f"{doc}")
