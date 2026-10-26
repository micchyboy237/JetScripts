import argparse
import os
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.chunking_utils import (
    chunk_markdown_hierarchy_with_data,
)
from jet.file.utils import load_file, save_file

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MD_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.md"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Chunk markdown text respecting header hierarchy and token limits.\n"
            "Outputs chunk results with full hierarchy metadata."
        )
    )
    parser.add_argument(
        "md_file",
        nargs="?",
        default=DEFAULT_MD_FILE,
        help="Path to the markdown file",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)",
    )
    parser.add_argument(
        "-s",
        "--overlap_strategy",
        choices=["sentence", "token", "none"],
        default="sentence",
        help='How to bridge adjacent chunks ("sentence", "token", "none")',
    )
    parser.add_argument(
        "-o",
        "--overlap_size",
        type=int,
        default=1,
        help="Number of sentences/tokens to carry to next chunk, if overlap enabled (default: 1)",
    )
    parser.add_argument(
        "-m",
        "--min_chunk_size",
        type=int,
        default=32,
        help="Minimum tokens for a chunk to be kept standalone (default: 32)",
    )
    parser.add_argument(
        "-b",
        "--buffer",
        type=int,
        default=0,
        help="Reserved token space to avoid exceeding chunk_size (default: 0)",
    )
    parser.add_argument(
        "-q",
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Do not show progress bar",
    )
    args = parser.parse_args()

    input_md_path = args.md_file
    if not os.path.isfile(input_md_path):
        raise FileNotFoundError(f"Markdown file not found: {input_md_path}")

    markdown_text = load_file(input_md_path)
    save_file(markdown_text, OUTPUT_DIR / "source.md")

    # --- Actual chunking using chunk_markdown_hierarchy_with_data ---
    results = chunk_markdown_hierarchy_with_data(
        markdown_text=markdown_text,
        chunk_size=args.chunk_size,
        overlap_strategy=args.overlap_strategy,
        overlap_size=args.overlap_size,
        min_chunk_size=args.min_chunk_size,
        buffer=args.buffer,
        show_progress=args.show_progress,
    )
    # results: List[dict], each with keys from MarkdownChunkResult

    # Save main output: all chunks with metadata
    save_file(results, OUTPUT_DIR / "chunks_with_metadata.json")

    # Save just the chunk contents (plain text only)
    chunk_texts = [r["content"] for r in results]
    save_file(chunk_texts, OUTPUT_DIR / "chunks.txt")

    # Save per-chunk headers and token counts
    chunk_headers = [
        {
            "header": r.get("header"),
            "parent_header": r.get("parent_header"),
            "level": r.get("level"),
            "num_tokens": r.get("num_tokens"),
        }
        for r in results
    ]
    save_file(chunk_headers, OUTPUT_DIR / "chunk_headers.json")

    print(
        f"Chunked {input_md_path} into {len(results)} chunks.\n"
        f"Results saved in: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
