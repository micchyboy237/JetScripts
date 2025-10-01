import os
import shutil
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.vectors.semantic_search.vector_search_simple import VectorSearch

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Real-world demonstration
if __name__ == "__main__":
    # 1. Specify preffered dimensions
    dimensions = None
    model_name: OLLAMA_MODEL_NAMES = "embeddinggemma"
    # Same example queries
    queries = [
        "Which planet is known as the Red Planet?",
    ]
    sample_docs = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    ]

    search_engine = VectorSearch(model_name, truncate_dim=dimensions)
    search_engine.add_documents(sample_docs)

    for query in queries:
        results = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for num, (doc, score) in enumerate(results, 1):
            print(f"\n{colorize_log(f"{num}.", "ORANGE")} (Score: {
                  colorize_log(f"{score:.3f}", "SUCCESS")})")
            print(f"{doc}")

    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")
