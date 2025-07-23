from jet.vectors.semantic_search.vector_search_with_spellchecker import SpellCorrectedSearchEngine


def main():
    # Sample documents with misspellings
    documents = [
        {"id": 1, "content": "The quick brown foxx jumps over the lazy dog"},
        {"id": 2, "content": "A beautifull garden blooms with collorful flowers"},
        {"id": 3, "content": "Teh sun sets slowly behind the mountan"},
    ]
    keywords = ["beautiful garden", "quick fox", "sunset mountain"]

    # Initialize search engine
    search_engine = SpellCorrectedSearchEngine()
    search_engine.add_documents(documents, keywords)

    # Perform searches
    for keyword in keywords:
        results = search_engine.search(keyword)
        print(f"\nSearch query: {keyword}")
        for result in results:
            print(
                f"Document {result['id']}: {result['content']} (Score: {result['score']:.2f})")

    # Print corrections for debugging
    for doc in documents:
        corrections = search_engine.get_corrections(doc["id"])
        print(f"\nCorrections for document {doc['id']}:")
        for correction in corrections:
            print(
                f"Original: {correction['original']} -> Corrected: {correction['corrected']}")


if __name__ == "__main__":
    main()
