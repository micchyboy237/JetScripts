"""Production-ready Markdown chunking helper with per-example output dirs."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from chonkie import Chunk, CodeChunker, MarkdownChef, RecursiveChunker
from rich.console import Console
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class MarkdownChunkResult:
    chunks: List[Chunk]
    tables: List = field(default_factory=list)
    code_blocks: List = field(default_factory=list)
    images: List = field(default_factory=list)
    full_content: str = ""


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------
def chunk_markdown(
    text: str,
    chunk_size: int = 512,
    lang: str = "en",
    keep_temp_file: bool = False,
) -> MarkdownChunkResult:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)

    try:
        chef = MarkdownChef()
        md_doc = chef.process(tmp_path)

        chunker = RecursiveChunker.from_recipe(
            "markdown", lang=lang, chunk_size=chunk_size
        )
        chunks = chunker.chunk(md_doc.content)

        return MarkdownChunkResult(
            chunks=chunks,
            tables=md_doc.tables,
            code_blocks=md_doc.code,
            images=md_doc.images,
            full_content=md_doc.content,
        )
    finally:
        if not keep_temp_file and tmp_path.exists():
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_results(result: MarkdownChunkResult, example_name: str) -> Path:
    example_dir = OUTPUT_DIR / example_name
    example_dir.mkdir(parents=True, exist_ok=True)

    chunks_data = []
    for i, chunk in enumerate(result.chunks, 1):
        (example_dir / f"chunk_{i:02d}.txt").write_text(chunk.text, encoding="utf-8")
        chunks_data.append(
            {
                "index": i,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
            }
        )

    (example_dir / "chunks.json").write_text(
        json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for i, table in enumerate(result.tables, 1):
        (example_dir / f"table_{i:02d}.md").write_text(table.content, encoding="utf-8")

    for i, code in enumerate(result.code_blocks, 1):
        lang = code.language or "txt"
        (example_dir / f"code_{i:02d}.{lang}").write_text(
            code.content, encoding="utf-8"
        )

    (example_dir / "full_content.md").write_text(result.full_content, encoding="utf-8")

    summary = {
        "num_chunks": len(result.chunks),
        "num_tables": len(result.tables),
        "num_code_blocks": len(result.code_blocks),
        "num_images": len(result.images),
        "chunk_token_counts": [c.token_count for c in result.chunks],
    }
    (example_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return example_dir


# ---------------------------------------------------------------------------
# Sample input
# ---------------------------------------------------------------------------
SAMPLE_MD = """
# Project Overview

This document describes the main features of our system.

## Installation

Install with:

```bash
pip install mypackage
```

## Features

| Feature       | Status      | Notes                  |
|---------------|-------------|------------------------|
| Fast search   | Done        | Uses vector index      |
| Batch upload  | In progress | Coming in v2.1         |
| Auth          | Done        | OAuth2 + API keys      |

## Code Example

Here is a simple Python helper:

```python
def process_data(items):
    results = []
    for item in items:
        if item.is_valid():
            results.append(item.transform())
    return results
```

## Conclusion

The system is ready for production use.
"""


# ---------------------------------------------------------------------------
# Expected values
# ---------------------------------------------------------------------------
EXPECTED_NUM_CHUNKS = 2
EXPECTED_NUM_TABLES = 1
EXPECTED_NUM_CODE_BLOCKS = 2
EXPECTED_NUM_IMAGES = 0
EXPECTED_CHUNK_1_TOKENS_MIN = 400
EXPECTED_CHUNK_1_TOKENS_MAX = 480
EXPECTED_CHUNK_2_TOKENS_MIN = 250
EXPECTED_CHUNK_2_TOKENS_MAX = 300

EXPECTED_PHRASES_CHUNK_1 = [
    "# Project Overview",
    "## Installation",
    "pip install mypackage",
    "## Features",
    "Fast search",
]
EXPECTED_PHRASES_CHUNK_2 = [
    "## Code Example",
    "def process_data",
    "## Conclusion",
]


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
def example_a_code_chunker_large() -> Path:
    console.rule("[bold]Example A – CodeChunker (chunk_size=2048)[/bold]")

    chunker = CodeChunker(
        language="markdown",
        tokenizer="character",
        chunk_size=2048,
        include_nodes=False,
    )
    chunks = chunker.chunk(SAMPLE_MD)
    result = MarkdownChunkResult(chunks=chunks, full_content=SAMPLE_MD)
    return save_results(result, "example_a_code_chunker_large")


def example_b_recursive_recipe() -> Path:
    console.rule("[bold]Example B – RecursiveChunker.from_recipe('markdown')[/bold]")

    chunker = RecursiveChunker.from_recipe("markdown", lang="en", chunk_size=512)
    chunks = chunker.chunk(SAMPLE_MD)
    result = MarkdownChunkResult(chunks=chunks, full_content=SAMPLE_MD)
    return save_results(result, "example_b_recursive_recipe")


def example_c_improved_markdown_chef() -> Path:
    console.rule("[bold]Example C (Improved) – MarkdownChef + full content[/bold]")

    result = chunk_markdown(SAMPLE_MD, chunk_size=512)

    assert len(result.chunks) == EXPECTED_NUM_CHUNKS
    assert len(result.tables) == EXPECTED_NUM_TABLES
    assert len(result.code_blocks) == EXPECTED_NUM_CODE_BLOCKS
    assert len(result.images) == EXPECTED_NUM_IMAGES

    c1, c2 = result.chunks[0], result.chunks[1]
    assert EXPECTED_CHUNK_1_TOKENS_MIN <= c1.token_count <= EXPECTED_CHUNK_1_TOKENS_MAX
    assert EXPECTED_CHUNK_2_TOKENS_MIN <= c2.token_count <= EXPECTED_CHUNK_2_TOKENS_MAX

    for phrase in EXPECTED_PHRASES_CHUNK_1:
        assert phrase in c1.text
    for phrase in EXPECTED_PHRASES_CHUNK_2:
        assert phrase in c2.text
    for chunk in result.chunks:
        assert chunk.text.strip()

    console.print("[bold green]All assertions passed![/bold green]")
    return save_results(result, "example_c_improved_markdown_chef")


def example_d_code_chunker_moderate() -> Path:
    console.rule("[bold]Example D – CodeChunker (chunk_size=600)[/bold]")

    chunker = CodeChunker(
        language="markdown",
        tokenizer="character",
        chunk_size=600,
        include_nodes=False,
    )
    chunks = chunker.chunk(SAMPLE_MD)
    result = MarkdownChunkResult(chunks=chunks, full_content=SAMPLE_MD)
    return save_results(result, "example_d_code_chunker_moderate")


# ---------------------------------------------------------------------------
# Show saved files as resource links (base name only)
# ---------------------------------------------------------------------------
def show_saved_files(example_dirs: List[Path]) -> None:
    console.print("\n")
    console.rule("[bold cyan]All Saved Files[/bold cyan]")

    for example_dir in example_dirs:
        console.print(f"\n[bold]{example_dir.name}/[/bold]")
        files = sorted(example_dir.glob("*"))
        for f in files:
            # Resource link using only the base name
            console.print(f"  • [link={f.as_uri()}]{f.name}[/link]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    console.print(Panel.fit("[bold magenta]Production Markdown Chunker[/bold magenta]"))

    example_dirs = [
        example_a_code_chunker_large(),
        example_b_recursive_recipe(),
        example_c_improved_markdown_chef(),
        example_d_code_chunker_moderate(),
    ]

    show_saved_files(example_dirs)


if __name__ == "__main__":
    main()
