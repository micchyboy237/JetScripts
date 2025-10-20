from jet.libs.bertopic.rag_bertopic import TopicRAG

def run_example_rag_bertopic():
    """Demonstrates RAG retrieval with varying document sets and queries."""

    # Example 1: Small doc set (short texts)
    docs_short = [
        "AI is transforming healthcare with diagnostic tools.",
        "Machine learning enhances medical imaging.",
        "Cats are great pets with playful behavior.",
    ]
    rag1 = TopicRAG(verbose=True)
    rag1.fit_topics(docs_short)
    print("\n[Example 1] Query: 'AI in medicine'")
    for r in rag1.retrieve_for_query("AI in medicine"):
        print(r)

    # Example 2: Medium doc set (mixed topics)
    docs_medium = [
        "Climate change affects polar bears and sea levels.",
        "Solar energy and wind power are sustainable options.",
        "The stock market fluctuates due to economic factors.",
        "Investors are optimistic about renewable energy stocks.",
        "Cooking pasta perfectly requires timing and temperature.",
        "Italian cuisine includes pasta, pizza, and olive oil.",
    ]
    rag2 = TopicRAG(verbose=True)
    rag2.fit_topics(docs_medium)
    print("\n[Example 2] Query: 'renewable energy investments'")
    for r in rag2.retrieve_for_query("renewable energy investments", top_topics=2, top_k=3):
        print(r)

    # Example 3: Large doc set (varied domains)
    docs_large = [
        "Quantum computing uses qubits for advanced computation.",
        "Artificial intelligence drives innovation across industries.",
        "Data privacy is critical for user trust in digital platforms.",
        "The history of the Roman Empire spans centuries of conquest.",
        "World War II changed global politics and alliances.",
        "Gardening improves mental health and provides fresh produce.",
        "Python programming is widely used in machine learning.",
        "Software testing ensures system reliability and performance.",
        "Cooking and nutrition are key to a healthy lifestyle.",
        "Electric vehicles are reducing carbon emissions worldwide.",
    ]
    rag3 = TopicRAG(verbose=True)
    rag3.fit_topics(docs_large)
    print("\n[Example 3] Query: 'machine learning and AI applications'")
    for r in rag3.retrieve_for_query("machine learning and AI applications", top_topics=3, top_k=5):
        print(r)

    # Example 4: Query with no expected match
    print("\n[Example 4] Query: 'medieval knights and castles'")
    results = rag3.retrieve_for_query("medieval knights and castles", top_topics=2, top_k=3)
    print("No relevant documents found." if not results else results)


if __name__ == "__main__":
    run_example_rag_bertopic()
