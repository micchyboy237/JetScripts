from typing import List, TypedDict
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.llm.mlx.tasks.search_contexts_by_index import search_contexts_by_index, SearchResult
from jet.vectors.hybrid_reranker import Models, ScoreResults, SearchResults, calculate_scores, load_models, search_documents
import json
import os

MODEL_PATH: str = "llama-3.2-3b-instruct-4bit"


class RerankResult(TypedDict):
    rank: int
    doc_idx: int
    distance: float
    raw_score: float
    normalized_score: float
    content: str


def hybrid_rerank(query: str, documents: list[str], top_k=10) -> List[RerankResult]:
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
            f"Doc {result['doc_idx']}: {json.dumps(result['content'])[:200]}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Raw Score: {result['raw_score']:.4f}")
        logger.success(f"  Normalized Score: {result['normalized_score']:.4f}")
    return final_results


if __name__ == "__main__":
    query = "List trending isekai reincarnation anime this year."
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_split_header_docs/searched_html_myanimelist_net_Isekai/contexts.json"
    top_k = 10
    top_n = 5
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    documents: List[str] = load_file(data_path)
    reranked_results = hybrid_rerank(query, documents, top_k=top_k)
    save_file({
        "query": query,
        "results": reranked_results
    }, f"{output_dir}/rerank-results.json")
    reranked_documents = [result["content"] for result in reranked_results]
    print(f"\nPerforming LLM-based context search for query: '{query}'")
    search_result: SearchResult = search_contexts_by_index(
        query, reranked_documents, MODEL_PATH, top_n=top_n
    )
    if not search_result["is_valid"]:
        logger.error(f"LLM search failed: {search_result['error']}")
        final_results = []
    else:
        print("\nTop ranked contexts:")
        final_results = []
        for rank_idx, res in enumerate(search_result["results"], start=1):
            final_results.append({
                "rank": rank_idx,
                "doc_idx": res["doc_idx"],
                "score": res["score"],
                "content": reranked_documents[res["doc_idx"]]
            })
            logger.debug(
                f"Assigned score for doc_idx {res['doc_idx']}: {res['score']}")
        for res in final_results[:5]:
            logger.newline()
            logger.debug(
                f"Doc {res['doc_idx']}: {json.dumps(res['content'])[:200]}")
            print(f"  Score: {res['score']:.4f}")
            logger.success(f"  Rank: {res['rank']}")
    save_file({
        "query": query,
        "results": final_results
    }, f"{output_dir}/search-results.json")
