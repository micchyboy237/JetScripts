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


def main():
    """Main function to demonstrate file search."""
    query = "MLX Autogen"
    directories = [
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects",
        # "/Users/jethroestrada/Desktop/External_Projects/AI",
    ]

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
    }, f"{output_dir}/results_{'split' if split_chunks else 'merged'}.json")

    split_chunks = False
    without_split_chunks_results = merge_results(
        with_split_chunks_results, tokenizer=count_tokens)
    print_results(query, without_split_chunks_results, split_chunks)
    save_file({
        "query": query,
        "count": len(without_split_chunks_results),
        "merged": not split_chunks,
        "results": without_split_chunks_results
    }, f"{output_dir}/results_{'split' if split_chunks else 'merged'}.json")


if __name__ == "__main__":
    main()
