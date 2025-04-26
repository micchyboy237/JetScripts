
from jet.vectors.hybrid_search_engine import HybridSearchEngine


if __name__ == "__main__":
    docs = [
        "How to cook delicious pasta at home",
        "The art of boiling noodles perfectly",
        "Best practices for making pasta",
        "Top 10 pasta recipes you must try",
        "Healthy salad recipes with vegetables",
    ]
    query = "noodle cooking tips"

    engine = HybridSearchEngine()
    engine.fit(docs)

    results = engine.search(query, top_n=5, alpha=0.5)

    print("\n🔎 Hybrid Search Results w/ Diversity:\n")
    for r in results:
        print(f"Score: {r['score']:.4f} | Document: {r['document']}")

    results = engine.search(query, top_n=5, alpha=0.5, diversity=False)

    print("\n🔎 Hybrid Search Results w/o Diversity:\n")
    for r in results:
        print(f"Score: {r['score']:.4f} | Document: {r['document']}")
