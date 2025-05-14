import json
from typing import List
from jet.file.utils import load_file
from jet.logger import logger
from jet.vectors.hybrid_reranker import Models, ScoreResults, SearchResults, calculate_scores, load_models, search_documents


if __name__ == "__main__":
    documents: List[str] = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_split_header_docs/contexts.json")

    query = "List trending isekai anime this year."
    top_k = 5

    print("Loading models...")
    models: Models = load_models()

    print(f"\nPerforming search for query: '{query}'")
    results: SearchResults = search_documents(
        query, documents, models["embedder"], models["cross_encoder"], k=top_k
    )

    print("\nInitial retrieval results:")
    for idx, dist in zip(results["candidate_indices"], results["candidate_distances"]):
        print(
            f"Doc {idx}: {json.dumps(documents[idx])[:200]} (Distance: {dist:.4f})")

    print("\nReranked results:")
    for idx, score in zip(results["reranked_indices"], results["reranked_scores"]):
        print(
            f"Doc {idx}: {json.dumps(documents[idx])[:200]} (Score: {score:.4f})")

    print("\nCombined scores:")
    score_results: ScoreResults = calculate_scores(query, documents, results)
    for idx, dist, raw_score, norm_score in zip(
        score_results["indices"],
        score_results["distances"],
        score_results["raw_scores"],
        score_results["normalized_scores"]
    ):
        logger.newline()
        logger.debug(f"Doc {idx}: {json.dumps(documents[idx])[:200]}")
        print(f"  Distance: {dist:.4f}")
        print(f"  Raw Score: {raw_score:.4f}")
        logger.success(f"  Normalized Score: {norm_score:.4f}")
