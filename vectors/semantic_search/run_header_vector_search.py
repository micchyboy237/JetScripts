import os
import shutil
from typing import List
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.vectors.semantic_search.header_vector_search import HeaderSearchResult, search_headers, merge_results


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def save_results(query: str, results: List[HeaderSearchResult], merge_chunks: bool):
    print(f"Search results for '{query}' in these dirs:")
    for num, result in enumerate(results[:10], start=1):
        header = result["header"]
        parent_header = result["parent_header"]
        start_idx = result["metadata"]["start_idx"]
        end_idx = result["metadata"]["end_idx"]
        chunk_idx = result["metadata"]["chunk_idx"]
        num_tokens = result["metadata"]["num_tokens"]
        score = result["score"]
        print(
            f"{colorize_log(f"{num}.)", "ORANGE")} Score: {colorize_log(f'{score:.3f}', 'SUCCESS')} | Chunk: {chunk_idx} | Tokens: {num_tokens} | Start - End: {start_idx} - {end_idx}\nParent: {parent_header} | Header: {header}")


def main():
    """Main function to demonstrate file search."""
    # Example usage
    query = "top rag strategies reddit 2025"
    embed_model: EmbedModelType = "static-retrieval-mrl-en-v1"
    llm_model: LLMModelType = "qwen3-1.7b-4bit"

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_4/top_rag_strategies_reddit_2025/docs.json"
    # Recursively collect all .html files under docs_file
    header_docs: List[HeaderDoc] = load_file(docs_file)["documents"]

    top_k = None
    threshold = 0.0  # Using default threshold
    chunk_size = 200
    chunk_overlap = 50

    merge_chunks = False
    without_merge_chunks_results = list(
        search_headers(
            header_docs,
            query,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            merge_chunks=merge_chunks,
            tokenizer_model=embed_model
        )
    )
    save_file({
        "query": query,
        "count": len(without_merge_chunks_results),
        "merged": merge_chunks,
        "results": without_merge_chunks_results
    }, f"{OUTPUT_DIR}/results_{'merged' if merge_chunks else 'split'}.json")

    merge_chunks = True
    with_merge_chunks_results = merge_results(without_merge_chunks_results)
    save_file({
        "query": query,
        "count": len(with_merge_chunks_results),
        "merged": merge_chunks,
        "results": with_merge_chunks_results
    }, f"{OUTPUT_DIR}/results_{'merged' if merge_chunks else 'split'}.json")


if __name__ == "__main__":
    main()
