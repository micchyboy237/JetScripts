import os
from typing import List
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.vectors.semantic_search.header_vector_search import HeaderSearchResult, search_headers


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


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

    save_file({
        "query": query,
        "count": len(results),
        "merged": merge_chunks,
        "results": results
    }, f"{OUTPUT_DIR}/results_{'merged' if merge_chunks else 'split'}.json")


def main():
    """Main function to demonstrate file search."""
    # Example usage
    query = "top rag strategies reddit 2025"
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit"

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_3/top_rag_strategies_reddit_2025/pages"
    # Recursively collect all .html files under docs_file
    html_files = []
    for root, dirs, files in os.walk(docs_file):
        for file in files:
            if file.lower().endswith('.html'):
                html_files.append(os.path.join(root, file))
    header_docs: List[HeaderDoc] = []
    for html in html_files:
        sub_header_docs = derive_by_header_hierarchy(html)
        header_docs.extend(sub_header_docs)

    save_file(header_docs, f"{OUTPUT_DIR}/header_docs.json")

    top_k = None
    threshold = 0.0  # Using default threshold
    chunk_size = 250
    chunk_overlap = 50
    tokenizer = get_tokenizer_fn(llm_model)

    def count_tokens(text):
        return len(tokenizer(text))

    merge_chunks = True
    with_merge_chunks_results = list(
        search_headers(
            header_docs,
            query,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            merge_chunks=merge_chunks,
            tokenizer=count_tokens
        )
    )
    save_results(query,  with_merge_chunks_results, merge_chunks)

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
            tokenizer=count_tokens
        )
    )
    save_results(query, without_merge_chunks_results, merge_chunks)


if __name__ == "__main__":
    main()
