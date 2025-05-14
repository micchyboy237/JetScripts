import json
from typing import List
from jet.file.utils import load_file
from jet.vectors.hybrid_reranker import SearchResults, load_models, search_documents


if __name__ == "__main__":
    documents: List[str] = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_split_header_docs/contexts.json")

    query = "List trending isekai anime this year."

    print("Loading models...")
    models = load_models()

    print(f"\nPerforming search for query: '{query}'")
    results: SearchResults = search_documents(
        query, documents, models["embedder"], models["cross_encoder"], k=5
    )

    print("\nInitial retrieval results:")
    for idx, dist in zip(results["candidate_indices"], results["candidate_distances"]):
        print(
            f"Doc {idx} | Distance: {dist:.4f} | {json.dumps(documents[idx])[:100]}")

    print("\nReranked results:")
    for idx, score in zip(results["reranked_indices"], results["reranked_scores"]):
        print(
            f"Doc {idx} | Score: {score:.4f} | {json.dumps(documents[idx])[:100]}")
