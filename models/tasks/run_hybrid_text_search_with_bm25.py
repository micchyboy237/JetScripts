import os
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.tasks.hybrid_text_search_with_bm25 import search_texts


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    query = "List all ongoing and upcoming isekai anime 2025."
    docs = load_file(docs_file)
    search_output = search_texts(
        query, docs, top_k=None, rerank_top_k=10)

    # Unpack the tuple if return_raw_scores is True
    results, raw_scores = search_output if isinstance(
        search_output, tuple) else (search_output, None)

    for result in results:
        logger.success(
            f"\nRank {result['rank']} (Document Index {result['doc_index']}):")
        print(f"Embedding Score: {result['embedding_score']:.4f}")
        print(f"Combined Score: {result['combined_score']:.4f}")
        print(f"Rerank Score: {result['score']:.4f}")
        print(f"Headers: {result['headers']}")
        print(f"Original Document:\n{result['text']}")

    save_file(results, f"{output_dir}/results.json")

    # Optionally log raw scores for debugging
    if raw_scores:
        logger.debug("Raw Scores: %s", raw_scores)
