
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

    print("\n🔎 Hybrid Search Results w/ MMR Diversity:\n")
    mmr_results = engine.search(
        query, top_n=5, alpha=0.5, use_mmr=True, lambda_param=0.7)
    for r in mmr_results:
        print(f"Score: {r['score']:.4f} | Document: {r['document']}")

    print("\n🔎 Hybrid Search Results w/o Diversity:\n")
    simple_results = engine.search(query, top_n=5, alpha=0.5, use_mmr=False)
    for r in simple_results:
        print(f"Score: {r['score']:.4f} | Document: {r['document']}")
