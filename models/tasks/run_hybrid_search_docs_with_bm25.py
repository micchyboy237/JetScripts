

from jet.file.utils import load_file
from jet.logger import logger
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    query = "List all ongoing and upcoming isekai anime 2025."
    docs = load_file(docs_file)
    results = search_docs(query, docs, rerank_top_k=10)
    for result in results:
        logger.success(
            f"\nRank {result['rank']} (Document Index {result['doc_index']}):")
        print(f"Embedding Score: {result['embedding_score']:.4f}")
        print(f"Combined Score: {result['combined_score']:.4f}")
        print(f"Rerank Score: {result['score']:.4f}")
        print(f"Headers: {result['headers']}")
        print(f"Original Document:\n{result['text']}")
