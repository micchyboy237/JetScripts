import argparse
import os
import shutil
from pathlib import Path
from typing import List
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.data.utils import generate_unique_id
from jet.file.utils import save_file
from jet.logger import logger
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.utils.file import group_by_base_dir, search_files
from jet.utils.language import detect_lang
from jet.utils.text import format_sub_dir
from jet.vectors.reranker.bm25 import rerank_bm25
from jet.vectors.semantic_search.file_vector_search import FileSearchResult, merge_results, search_files as search_files_vector


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main(query: str, search_dir: str, extensions: List[str], include_files: List[str], exclude_files: List[str], max_group_depth: int, embed_model: EmbedModelType = "static-retrieval-mrl-en-v1") -> None:
    """
    Run file vector search and save results to JSON files.

    Args:
        query: Search query string.
        search_dir: Directory to search in.
        extensions: File extensions to include.
        include_files: Files or directories to include.
        exclude_files: Files or directories to exclude.
        max_group_depth: Maximum depth for grouping directories.
    """
    output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"

    # Remove existing output directory
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    tokenizer = SentenceTransformerRegistry.get_tokenizer(embed_model)

    def count_tokens(text):
        return len(tokenizer.encode(text))

    def preprocess_text(text):
        return clean_markdown_links(text)

    # Search for files
    try:
        # Get top 50 results from all files
        all_results = list(
            search_files_vector(
                search_dir,
                query,
                extensions,
                top_k=50,
                threshold=0.5,
                embed_model=embed_model,
                chunk_size=256,
                chunk_overlap=40,
                split_chunks=True,
                tokenizer=count_tokens,
                preprocess=preprocess_text,
                includes=[f"**/{f}/*" for f in include_files],
                excludes=[f"**/{f}/*" for f in exclude_files],
                weights={
                    "dir": 0.325,
                    "name": 0.325,
                    "content": 0.35,
                }
            )
        )

        all_files = search_files(
            search_dir,
            extensions,
            include_files=include_files,
            exclude_files=exclude_files,
        )

        # Save all files
        save_file({
            "query": query,
            "count": len(all_files),
            "files": all_files
        }, f"{output_dir}/files.json", verbose=True)

        # Group files by directory
        grouped_dirs = group_by_base_dir(
            all_files, search_dir, max_depth=max_group_depth)

        # Save grouped directories
        save_file({
            "query": query,
            "count": len(grouped_dirs),
            "results": grouped_dirs
        }, f"{output_dir}/grouped_dirs.json", verbose=True)

        # Perform vector search and save results for each directory group
        for dir_group in grouped_dirs:
            base_dir = Path(search_dir) / dir_group
            base_name = base_dir.name
            top_k = None
            threshold = 0.0
            chunk_size = 512
            chunk_overlap = 80

            results = list(
                search_files_vector(
                    str(base_dir),
                    query,
                    extensions,
                    top_k=top_k,
                    threshold=threshold,
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    split_chunks=False,
                    tokenizer=count_tokens,
                    preprocess=preprocess_text,
                    includes=[f"**/{f}/*" for f in include_files],
                    excludes=[f"**/{f}/*" for f in exclude_files],
                    weights={
                        "dir": 0.325,
                        "name": 0.325,
                        "content": 0.35,
                    }
                )
            )

            filtered_results = [
                result for result in results
                if detect_lang(result["text"])["lang"] == "en"
            ]

            # Save search results for this directory group
            search_results_path = f"{output_dir}/{base_name}/search_results.json"
            save_file({
                "query": query,
                "count": len(filtered_results),
                "merged": False,  # Updated to reflect split_chunks=True
                "results": filtered_results
            }, search_results_path, verbose=True)

    except Exception as e:
        logger.error(f"Error in file search or saving: {str(e)}")
        raise


if __name__ == "__main__":
    query = "long context summary agent"
    search_dir = "/Users/jethroestrada/Desktop/External_Projects/AI"
    extensions = [".py"]
    include_files = ["examples"]
    exclude_files = [".venv", ".pytest_cache", "node_modules"]
    max_group_depth = 2
    # embed_model: EmbedModelType = "static-retrieval-mrl-en-v1"
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"

    main(query, search_dir, extensions, include_files,
         exclude_files, max_group_depth, embed_model)
