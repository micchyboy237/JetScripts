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
from rich.console import Console
from rich.table import Table
from rich.text import Text

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
CONTEXT_SAFETY_MARGIN = 32


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
        return count_tokens(text, model=embed_model)

    def preprocess_text(text):
        return clean_markdown_links(text)

    # Track saved files for summary
    saved_files = []

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
        # Sort all_results by rank in ascending order
        all_results.sort(key=lambda x: x.get("rank", float("inf")))

        all_results_path = f"{output_dir}/all_results.json"
        save_file(
            {
                "query": query,
                "count": len(all_results),
                "includes": include_files,
                "excludes": exclude_files,
                "results": all_results,
            },
            all_results_path,
            verbose=True,
        )
        saved_files.append(("All Results", all_results_path, len(all_results)))

        all_files = search_files(
            search_dir,
            extensions,
            include_files=include_files,
            exclude_files=exclude_files,
        )

        files_path = f"{output_dir}/files.json"
        save_file(
            {
                "query": query,
                "count": len(all_files),
                "includes": include_files,
                "excludes": exclude_files,
                "files": all_files,
            },
            files_path,
            verbose=True,
        )
        saved_files.append(("File List", files_path, len(all_files)))

        grouped_dirs = group_by_base_dir(
            all_files, search_dir, max_depth=max_group_depth
        )

        grouped_dirs_path = f"{output_dir}/grouped_dirs.json"
        save_file(
            {"query": query, "count": len(grouped_dirs), "groups": grouped_dirs},
            grouped_dirs_path,
            verbose=True,
        )
        saved_files.append(("Directory Groups", grouped_dirs_path, len(grouped_dirs)))

        for dir_group in grouped_dirs:
            base_dir = Path(search_dir) / dir_group
            base_name = base_dir.name
            top_k = None
            threshold = 0.0
            chunk_overlap = 80
            results = list(
                search_files_vector(
                    str(base_dir),
                    query,
                    extensions,
                    top_k=top_k,
                    threshold=threshold,
                    embed_model=embed_model,
                    chunk_size=512 - CONTEXT_SAFETY_MARGIN,
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
            # Sort filtered_results by rank in ascending order
            filtered_results.sort(key=lambda x: x.get("rank", float("inf")))

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
            saved_files.append(
                (f"Group: {base_name}", search_results_path, len(filtered_results))
            )

        # Display summary using rich console
        console = Console()
        console.print("\n")

        # Title
        title = Text("📊 Vector Search Results Summary", style="bold cyan")
        console.print(title)
        console.print("─" * 50, style="dim")

        # Query info
        console.print(f"🔍 Query: [bold yellow]{query}[/bold yellow]")
        console.print(
            f"📁 Output Directory: [bold blue]{Path(output_dir).name}[/bold blue]"
        )
        console.print("─" * 50, style="dim")

        # Create table for saved files
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan", width=25)
        table.add_column("File", style="green", width=30)
        table.add_column("Items", justify="right", style="yellow", width=10)

        for file_type, file_path, item_count in saved_files:
            # Shorten path to base name
            short_path = Path(file_path).name
            # Create clickable file link
            file_link = f"file://{file_path}"
            table.add_row(
                file_type, f"[link={file_link}]{short_path}[/link]", str(item_count)
            )

        console.print(table)
        console.print("─" * 50, style="dim")

        # Total summary
        total_items = sum(count for _, _, count in saved_files)
        console.print(
            f"✅ Total files saved: [bold green]{len(saved_files)}[/bold green]"
        )
        console.print(
            f"📈 Total items across all files: [bold green]{total_items}[/bold green]"
        )
        console.print("\n")

    except Exception as e:
        logger.error(f"Error in file search or saving: {str(e)}")
        raise


import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Run file vector search with options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="research multi agent",
        help="Search query",
    )
    parser.add_argument(
        "-s",
        "--search-dir",
        type=str,
        default="/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/en",
        help="Directory to search in",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        default=".md",
        help="Comma-separated list of file extensions to include (e.g. .md,.py)",
    )
    parser.add_argument(
        "-i",
        "--include",
        nargs="*",
        default=[],
        help="Files or subdirectories to include",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        nargs="*",
        default=[".venv", ".pytest_cache", "node_modules"],
        help="Files or subdirectories to exclude",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=2,
        help="Maximum group depth",
    )
    parser.add_argument(
        "-m",
        "--embed-model",
        type=str,
        default=EMBED_MODEL,
        help=f"Embedding model to use (default: {EMBED_MODEL})",
    )
    args = parser.parse_args()

    # Parse extensions as list
    args.extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]
    return args


if __name__ == "__main__":
    args = get_args()
    main(
        args.query,
        args.search_dir,
        args.extensions,
        args.include,
        args.exclude,
        args.depth,
        args.embed_model,
    )
