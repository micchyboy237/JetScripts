import os
from jet.file.utils import load_file, save_file
from jet.vectors.reranker.bm25 import rerank_bm25

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/vectors/semantic_search/generated/run_file_vector_search/results_merged.json"
    docs = load_file(docs_file)["results"][:50]

    query = "RAG agents"
    texts = [doc["code"] for doc in docs]

    query_candidates, reranked_results = rerank_bm25(
        query, texts, ids=[str(idx) for idx, _ in enumerate(texts)])

    save_file({
        "query": query,
        "candidates": query_candidates,
        "count": len(reranked_results),
        "results": reranked_results
    }, f"{OUTPUT_DIR}/example/reranked_results.json")
