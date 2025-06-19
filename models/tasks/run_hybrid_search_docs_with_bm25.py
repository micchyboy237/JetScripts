import os
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs
from jet.models.tokenizer.base import count_tokens
from jet.models.model_types import LLMModelType
from jet.vectors.document_types import HeaderDocument


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    docs = load_file(docs_file)
    query = docs["query"]
    docs = docs["documents"]
    docs = [HeaderDocument(**doc) for doc in docs]
    docs_to_search = [
        doc for doc in docs if doc.metadata["header_level"] != 1 and doc.metadata["content"].strip()]
    logger.debug(
        f"Filtered to {len(docs_to_search)} documents for search (excluding header level 1)")
    results = search_docs(
        query,
        documents=docs_to_search,
        ids=[doc.id_ for doc in docs_to_search],
        top_k=None,
    )

    logger.info(f"Counting tokens ({len(results)})...")
    token_counts: list[int] = count_tokens(
        llm_model, [result['text'] for result in results], prevent_total=True)

    for result, tokens in zip(results, token_counts):
        logger.success(
            f"\nRank {result['rank']} (Doc: {result['doc_index']} | Tokens: {tokens}):")
        print(f"Embedding Score: {result['embedding_score']:.4f}")
        print(f"Combined Score: {result['combined_score']:.4f}")
        print(f"Final Score: {result['score']:.4f}")
        print(f"Headers: {result['headers']}")
        print(f"Original Document:\n{result['text']}")

    results_no_node = [
        {k: v for k, v in result.items() if k != 'node'} for result in results]

    save_file({"query": query, "results": results_no_node},
              f"{output_dir}/results.json")
    save_file(token_counts, f"{output_dir}/tokens.json")
