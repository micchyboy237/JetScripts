"""
rag_cli.py
Command-line interface for RAG NLP pipeline demo.

Usage Examples:
---------------
# Run with a single markdown file
python rag_cli.py --query "Explain sliding window" --docs ./samples/example.md

# Run with a folder of markdown files
python rag_cli.py --query "What is MMR?" --docs ./samples/

# Enable topic tagging and markdown chunking
python rag_cli.py --query "topic clustering" --docs ./docs --with-topics
"""

import os
import shutil
import glob
from typing import List
from rich.console import Console
from rich.table import Table
from rich.progress import track

from jet.libs.stanza.rag_nlp import (
    RAGPipeline,
    tag_topics,
    Chunk,
)
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

console = Console()


# =============== FILE HANDLING ==================

def load_markdown_files(path: str) -> List[str]:
    """Load markdown text(s) from file or folder."""
    files = []
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**/*.md"), recursive=True)
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    texts = []
    for f in track(files, description=f"[cyan]Loading {len(files)} markdown files..."):
        with open(f, "r", encoding="utf-8") as fh:
            texts.append(fh.read())
    return texts


# =============== DISPLAY HELPERS ==================

def show_results(query: str, chunks: List[Chunk]):
    """Display retrieved chunks as a table and store row data."""
    table = Table(title=f"RAG Retrieval Results for Query: [green]{query}[/green]")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Section", justify="left", style="magenta")
    table.add_column("Chunk Text", justify="left", style="white", no_wrap=False)
    table.add_column("Topic", justify="center", style="yellow")
    table.add_column("Sim", justify="center", style="green")
    table.add_column("Div", justify="center", style="blue")
    row_data = []
    for idx, c in enumerate(chunks):
        text_preview = (c.text[:200] + "...") if len(c.text) > 200 else c.text
        topic = str(c.metadata.get("topic")) if c.metadata and "topic" in c.metadata else "-"
        section = c.section_title or "-"
        sim_score = f"{c.metadata.get('similarity', 0):.3f}" if c.metadata else "-"
        div_score = f"{c.metadata.get('diversity', 0):.3f}" if c.metadata and "diversity" in c.metadata else "-"
        table.add_row(str(idx + 1), section, text_preview, topic, sim_score, div_score)
        row_data.append({
            "Rank": str(idx + 1),
            "Section": section,
            "Chunk Text": text_preview,
            "Topic": topic,
            "Sim": sim_score,
            "Div": div_score
        })
    table.row_data = row_data  # Attach row data to table object
    console.print(table)
    return table


# =============== MAIN RUNNER ==================

def main():
    # Replace argparse with static variables
    query = "Top isekai anime 2025"
    docs = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/stanza/sample_docs"
    chunk_mode = "markdown"  # options: "markdown", "sentence"
    top_k = 10
    stride_ratio = 0.3
    diversity = 0.5
    use_mmr = False  # Set True to enable MMR-based retrieval
    with_topics = False  # Set True to enable BERTopic topic tagging
    debug = False  # Set True to print debug info
    class Args:
        pass

    args = Args()
    args.query = query
    args.docs = docs
    args.chunk_mode = chunk_mode
    args.top_k = top_k
    args.stride_ratio = stride_ratio
    args.diversity = diversity
    args.use_mmr = use_mmr
    args.with_topics = with_topics
    args.debug = debug

    save_file(args, f"{OUTPUT_DIR}/inputs.json")

    console.print("[bold green]>>> Loading RAG pipeline...[/bold green]")
    pipeline = RAGPipeline(
        model_name="all-MiniLM-L6-v2",
        use_markdown=(args.chunk_mode == "markdown"),
        stride_ratio=args.stride_ratio,
    )

    console.print(f"[bold blue]>>> Reading documents from:[/bold blue] {args.docs}")
    docs = load_markdown_files(args.docs)

    all_chunks: List[Chunk] = []
    for doc in docs:
        chunks = pipeline.prepare_chunks(doc)
        all_chunks.extend(chunks)
    console.print(f"[bold cyan]>>> Created {len(all_chunks)} chunks[/bold cyan]")
    save_file([chunk.to_dict() for chunk in all_chunks], f"{OUTPUT_DIR}/all_chunks.json")

    if args.with_topics:
        console.print("[bold yellow]>>> Fitting BERTopic model for topic tagging...[/bold yellow]")
        topic_model = tag_topics(all_chunks, pipeline.embedder)
        console.print(f"[green]Found {len(set([c.metadata['topic'] for c in all_chunks]))} topics[/green]")

    console.print(f"[bold blue]>>> Running {'MMR' if args.use_mmr else 'similarity'} retrieval for query:[/bold blue] '{args.query}'")
    results = pipeline.retrieve(
        args.query,
        all_chunks,
        top_k=args.top_k,
        diversity=args.diversity if args.use_mmr else None,
    )
    results_table = show_results(args.query, results)
    save_file([result.to_dict() for result in results], f"{OUTPUT_DIR}/search_results.json")

    if args.debug:
        for idx, c in enumerate(results):
            console.print(f"\n[bold magenta]--- Chunk {idx+1} ---[/bold magenta]")
            console.print(f"[cyan]Section:[/cyan] {c.section_title}")
            console.print(f"[yellow]Tokens:[/yellow] {len(c.text.split())}")
            console.print(f"[white]{c.text}[/white]\n")

    save_file(results_table, f"{OUTPUT_DIR}/results.json")
    save_file(results_table, f"{OUTPUT_DIR}/table.md")


if __name__ == "__main__":
    main()
