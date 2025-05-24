import json
import os
from typing import List
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.vectors.hybrid_reranker import Models, SearchResult, load_models, search_documents

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    documents: List[str] = load_file(docs_file)
    query = "List trending isekai anime this year.\nList trending isekai anime around 2025.\nIdentify popular isekai anime titles that have gained significant attention in recent times.\nProvide a brief description or highlight key features of the most trending isekai anime this year."

    top_k = None

    print("Loading models...")
    models: Models = load_models()

    print(f"\nPerforming search for query: '{query}'")
    final_results: List[SearchResult] = search_documents(
        query, documents, models["embedder"], models["cross_encoder"], k=top_k
    )

    print("\nCombined scores (sorted by raw score descending):")
    for result in final_results[:5]:
        logger.newline()
        logger.debug(
            f"Doc {result['doc_idx']}: {json.dumps(result['content'])[:200]}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Raw Score: {result['raw_score']:.4f}")
        logger.success(f"  Normalized Score: {result['normalized_score']:.4f}")

    save_file({
        "query": query,
        "results": final_results
    }, f"{output_dir}/rerank-results.json")
