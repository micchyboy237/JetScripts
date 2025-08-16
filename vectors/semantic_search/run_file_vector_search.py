import argparse
import os
from typing import List
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.utils.language import detect_lang
from jet.utils.text import format_sub_dir
from jet.vectors.reranker.bm25 import rerank_bm25
from jet.vectors.semantic_search.file_vector_search import FileSearchResult, merge_results, search_files


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def print_results(query: str, results: List[FileSearchResult], split_chunks: bool):
    for num, result in enumerate(results[:10], start=1):
        file_path = result["metadata"]["file_path"]
        start_idx = result["metadata"]["start_idx"]
        end_idx = result["metadata"]["end_idx"]
        chunk_idx = result["metadata"]["chunk_idx"]
        num_tokens = result["metadata"]["num_tokens"]
        score = result["score"]
        print(
            f"{colorize_log(f"{num}.)", "ORANGE")} Score: {colorize_log(f'{score:.3f}', 'SUCCESS')} | Chunk: {chunk_idx} | Tokens: {num_tokens} | Start - End: {start_idx} - {end_idx}\nFile: {file_path}")


def rerank_results(query: str, results: List[FileSearchResult]):
    texts = [result["text"] for result in results]
    ids = [str(idx) for idx, _ in enumerate(texts)]
    metadatas = [result["metadata"] for result in results]

    query_candidates, reranked_results = rerank_bm25(
        query, texts, ids=ids, metadatas=metadatas)

    return query_candidates, reranked_results


def main(query, directories):
    """Main function to demonstrate file search."""
    output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"

    extensions = [".py"]
    embed_model: EmbedModelType = "static-retrieval-mrl-en-v1"
    llm_model: LLMModelType = "qwen3-1.7b-4bit"

    top_k = None
    threshold = 0.0  # Using default threshold
    chunk_size = 1000
    chunk_overlap = 100
    tokenizer = SentenceTransformerRegistry.get_tokenizer(embed_model)

    def count_tokens(text):
        return len(tokenizer.encode(text))

    def preprocess_text(text):
        return clean_markdown_links(text)

    split_chunks = True

    print(f"Search results for '{query}' in these dirs:")
    for d in directories:
        print(d)

    with_split_chunks_results = list(
        search_files(
            directories, query, extensions,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_chunks=split_chunks,
            tokenizer=count_tokens,
            preprocess=preprocess_text,
            excludes=["**/.venv/*", "**/.pytest_cache/*", "**/node_modules/*"],
            weights={
                "dir": 0.325,
                "name": 0.325,
                "content": 0.35,
            }
        )
    )
    with_split_chunks_results = [
        result for result in with_split_chunks_results
        if detect_lang(result["text"])["lang"] == "en"
    ]
    print_results(query, with_split_chunks_results, split_chunks)
    save_file({
        "query": query,
        "count": len(with_split_chunks_results),
        "merged": not split_chunks,
        "results": with_split_chunks_results
    }, f"{output_dir}/search_results_split.json")

    split_chunks = False
    without_split_chunks_results = merge_results(
        with_split_chunks_results, tokenizer=count_tokens)
    print_results(query, without_split_chunks_results, split_chunks)
    save_file({
        "query": query,
        "count": len(without_split_chunks_results),
        "merged": not split_chunks,
        "results": without_split_chunks_results
    }, f"{output_dir}/search_results_merged.json")

    # Rerank

    query_candidates, reranked_results = rerank_results(
        query, with_split_chunks_results)
    save_file({
        "query": query,
        "candidates": query_candidates,
        "count": len(reranked_results),
        "results": reranked_results
    }, f"{output_dir}/example/reranked_results_split.json")

    query_candidates, reranked_results = rerank_results(
        query, without_split_chunks_results)
    save_file({
        "query": query,
        "candidates": query_candidates,
        "count": len(reranked_results),
        "results": reranked_results
    }, f"{output_dir}/example/reranked_results_merged.json")


def parse_arguments():
    """Parse command line arguments for query and directories.

    Usage:
        Positional: python file_search.py "query" /path1 /path2
        Named: python file_search.py --query "query" --directories /path1 /path2
        Mixed: python file_search.py "query" --directories /path1
        Default: python file_search.py

    Sample Commands:
        1. python file_search.py "neural network" /Users/jethroestrada/Desktop/AI/mlx
        2. python file_search.py --query "embedding model" --directories /Users/jethroestrada/Desktop/AI/mlx
        3. python file_search.py "function call" /Users/jethroestrada/Desktop/AI/mlx-lm

    Notes:
        - Defaults to "MLX Tools or Function Call" and predefined directories if not provided.
        - Positional args must precede named args.
    """
    parser = argparse.ArgumentParser(
        description="File search with query and directories")
    parser.add_argument("query", type=str, nargs="?",
                        default="MLX Tools or Function Call", help="Search query")
    parser.add_argument("directories", type=str, nargs="*", default=[
        "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/mlx",
        "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/mlx-lm",
        "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/mlx-embeddings",
        "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/mlx-vlm",
        "/Users/jethroestrada/Desktop/External_Projects/AI/examples_05_2025/mlx-examples",
    ], help="Search directories")
    parser.add_argument("--query", type=str, dest="query_flag",
                        default=None, help="Alternative query input")
    parser.add_argument("--directories", type=str, nargs="+", dest="directories_flag",
                        default=None, help="Alternative directories input")

    args = parser.parse_args()
    query = args.query_flag if args.query_flag is not None else args.query
    directories = args.directories_flag if args.directories_flag is not None else args.directories

    return argparse.Namespace(query=query, directories=directories)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.query, args.directories)
