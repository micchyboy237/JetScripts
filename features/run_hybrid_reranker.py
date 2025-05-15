import json
import os
from typing import List
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.vectors.hybrid_reranker import Models, ScoreResults, SearchResults, calculate_scores, load_models, search_documents


if __name__ == "__main__":
    documents: List[str] = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_split_header_docs/searched_html_myanimelist_net_Isekai/contexts.json")
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    query = "List trending isekai anime this year."
    top_k = None

    print("Loading models...")
    models: Models = load_models()

    print(f"\nPerforming search for query: '{query}'")
    results: SearchResults = search_documents(
        query, documents, models["embedder"], models["cross_encoder"], k=top_k
    )

    print("\nCombined scores (sorted by raw score descending):")
    final_results = []
    score_results: ScoreResults = calculate_scores(query, documents, results)
    for rank_idx, (idx, dist, raw_score, norm_score) in enumerate(list(zip(
        score_results["indices"],
        score_results["distances"],
        score_results["raw_scores"],
        score_results["normalized_scores"]
    )), start=1):
        final_results.append({
            "rank": rank_idx,
            "doc_idx": idx,
            "distance": dist,
            "raw_score": raw_score,
            "normalized_score": norm_score,
            "content": documents[idx]
        })

    for result in final_results[:5]:
        logger.newline()
        logger.debug(
            f"Doc {result["doc_idx"]}: {json.dumps(result["content"])[:200]}")
        print(f"  Distance: {result["distance"]:.4f}")
        print(f"  Raw Score: {result["raw_score"]:.4f}")
        logger.success(f"  Normalized Score: {result["normalized_score"]:.4f}")

    save_file({
        "query": query,
        "results": final_results
    }, f"{output_dir}/rerank-results.json")
