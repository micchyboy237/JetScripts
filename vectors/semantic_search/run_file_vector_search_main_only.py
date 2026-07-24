import os
import shutil
from pathlib import Path
from typing import List

from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.file.utils import save_file
from jet.logger import logger
from jet.utils.file import group_by_base_dir, search_files
from jet.utils.language import detect_lang
from jet.utils.text import format_sub_dir
from jet.vectors.semantic_search.file_vector_search import (
    search_files as search_files_vector,
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main(
    query: str,
    search_dir: str,
    extensions: List[str],
    include_files: List[str],
    exclude_files: List[str],
    max_group_depth: int,
    embed_model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
) -> None:
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
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    def get_tokens(text):
        return count_tokens(text)

    def preprocess_text(text):
        return clean_markdown_links(text)

    try:
        all_results = list(
            search_files_vector(
                search_dir,
                query,
                extensions,
                top_k=50,
                threshold=0.2,
                embed_model=embed_model,
                chunk_size=256,
                chunk_overlap=40,
                split_chunks=True,
                tokenizer=get_tokens,
                preprocess=preprocess_text,
                includes=[f"**/{f}/*" for f in include_files],
                excludes=[f"**/{f}/*" for f in exclude_files],
                weights={
                    "dir": 0.325,
                    "name": 0.325,
                    "content": 0.35,
                },
            )
        )

        save_file(
            {
                "query": query,
                "count": len(all_results),
                "includes": include_files,
                "excludes": exclude_files,
                "results": all_results,
            },
            f"{output_dir}/all_results.json",
            verbose=True,
        )

        all_files = search_files(
            search_dir,
            extensions,
            include_files=include_files,
            exclude_files=exclude_files,
        )

        save_file(
            {
                "query": query,
                "count": len(all_files),
                "includes": include_files,
                "excludes": exclude_files,
                "files": all_files,
            },
            f"{output_dir}/files.json",
            verbose=True,
        )

        grouped_dirs = group_by_base_dir(
            all_files, search_dir, max_depth=max_group_depth
        )

        save_file(
            {"query": query, "count": len(grouped_dirs), "groups": grouped_dirs},
            f"{output_dir}/grouped_dirs.json",
            verbose=True,
        )

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
                    tokenizer=get_tokens,
                    preprocess=preprocess_text,
                    includes=[f"**/{f}/*" for f in include_files],
                    excludes=[f"**/{f}/*" for f in exclude_files],
                    weights={
                        "dir": 0.325,
                        "name": 0.325,
                        "content": 0.35,
                    },
                )
            )

            filtered_results = [
                result
                for result in results
                if detect_lang(result["text"])["lang"] == "en"
            ]

            search_results_path = f"{output_dir}/{base_name}/search_results.json"
            save_file(
                {
                    "query": query,
                    "count": len(filtered_results),
                    "merged": False,
                    "results": filtered_results,
                },
                search_results_path,
                verbose=True,
            )

    except Exception as e:
        logger.error(f"Error in file search or saving: {str(e)}")
        raise


if __name__ == "__main__":
    query = "research multi agent"
    search_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/en"
    extensions = [".md"]
    # include_files = ["examples"]
    include_files = []
    exclude_files = [".venv", ".pytest_cache", "node_modules"]
    max_group_depth = 2
    embed_model: LLAMACPP_EMBED_KEYS = EMBED_MODEL

    main(
        query,
        search_dir,
        extensions,
        include_files,
        exclude_files,
        max_group_depth,
        embed_model,
    )
