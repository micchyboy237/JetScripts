import os
import shutil
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType
from jet.vectors.semantic_search.vector_search_simple import VectorSearch

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Real-world demonstration
if __name__ == "__main__":
    # 1. Specify preffered dimensions
    dimensions = None
    # dimensions = 512
    model_name: EmbedModelType = "mxbai-embed-large"
    # model_name: EmbedModelType = "nomic-embed-text"
    # model_name: EmbedModelType = "all-MiniLM-L6-v2"
    # Same example queries
    queries = [
        "Helo, wrld! I am fien.",
        "Helo, world! I am fien.",
        "Hello, wrld! I am fien.",
    ]

    search_engine = VectorSearch(model_name, truncate_dim=dimensions)

    # Same sample documents
    sample_docs = [
        "wold",
        "wild",
        "weld",
        "wald",
        "world",
    ]
    search_engine.add_documents(sample_docs)

    all_results = []
    for query in queries:
        candidates = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for num, (doc, score) in enumerate(candidates, 1):
            print(f"{colorize_log(f"{num}.", "ORANGE")} | Score: {
                  colorize_log(f"{score:.3f}", "SUCCESS")} | {doc[:50]}")
        all_results.append({
            "query": query,
            "candidates": candidates
        })

    # Print only the top result for each query
    print("\nTop result for each query:")
    for result in all_results:
        query = result["query"]
        candidates = result["candidates"]
        if candidates:
            top_doc, top_score = candidates[0]
            print(f"Query: {query}")
            print(f"  Top match: {colorize_log(f'{top_doc}', 'SUCCESS')} (Score: {
                  colorize_log(f'{top_score:.3f}', 'ORANGE')})")
        else:
            print(f"Query: {query}")
            print("  No matches found.")

    save_file({
        "model": model_name,
        "count": len(all_results),
        "results": all_results
    }, f"{OUTPUT_DIR}/results.json")
