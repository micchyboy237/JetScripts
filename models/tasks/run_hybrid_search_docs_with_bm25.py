import os
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs
from jet.models.tokenizer.base import count_tokens
from jet.models.model_types import LLMModelType


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    query = "List all ongoing and upcoming isekai anime 2025."
    docs = load_file(docs_file)
    search_output = search_docs(
        query,
        docs,
        top_k=20,
        return_raw_scores=True,
        with_bm25=False,
        with_rerank=False,
    )

    # Unpack the tuple if return_raw_scores is True
    results, raw_scores = search_output if isinstance(
        search_output, tuple) else (search_output, None)
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

    # Optionally log raw scores for debugging
    if raw_scores:
        logger.debug("Raw Scores: %s", raw_scores)

    results_no_document = [
        {k: v for k, v in result.items() if k != 'document'} for result in results]

    save_file(results_no_document, f"{output_dir}/results.json")
    save_file(token_counts, f"{output_dir}/tokens.json")
    save_file(raw_scores, f"{output_dir}/raw_scores.json")
