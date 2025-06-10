

from jet.logger import logger
from jet.models.tasks.hybrid_search_docs import search_docs


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    query = "List all ongoing and upcoming isekai anime 2025."
    results = search_docs(
        docs_file, query, rerank_top_k=10, embedder_model="all-MiniLM-L12-v2")
    for result in results:
        logger.success(
            f"\nRank {result['rank']} (Document ID {result['doc_id']}):")
        print(f"Embedding Score: {result['embedding_score']:.4f}")
        print(f"Combined Score: {result['combined_score']:.4f}")
        print(f"Rerank Score: {result['rerank_score']:.4f}")
        print(f"Headers: {result['headers']}")
        print(f"Original Document:\n{result['text']}")
