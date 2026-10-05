"""
04_demo_topic_docs_search.py
Enriched topic search returning similarity, keywords, and representative docs.
Saves comprehensive search report as CSV and JSON.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List

import pandas as pd
from jet.libs.bertopic.monkey_patches.add_check_array import init_patch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

init_patch()

from jet.adapters.bertopic.factory_with_repr import (
    create_bertopic_embedder,
    create_topic_model,
    find_topics_with_data,
    sanity_check_embedder,
)

# --- Output Directory Setup ---
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("rich")
console = Console()


def run_topic_docs_search_demo(
    documents: List[str],
    query: str,
    top_n: int = 5,
    max_reps: int = 2,
) -> pd.DataFrame:
    """Enriched topic search with representative documents."""

    # Save Inputs & Config
    with open(OUTPUT_DIR / "inputs.json", "w", encoding="utf-8") as f:
        json.dump(
            {"documents": documents, "query": query}, f, indent=2, ensure_ascii=False
        )

    config = {"top_n": top_n, "max_reps": max_reps, "min_topic_size": 2}
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Fit Model
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)
    topic_model = create_topic_model(embedder=embedder, min_topic_size=2, verbose=False)
    embeddings = embedder.embed(documents, verbose=False)
    _, _ = topic_model.fit_transform(documents, embeddings=embeddings)

    # Enriched Search
    logger.info(f"Running enriched search for: [yellow]'{query}'[/yellow]")
    rich_df = find_topics_with_data(
        topic_model,
        query,
        docs=documents,
        top_n=top_n,
        max_reps=max_reps,
        verbose=False,
    )

    # Save Results
    rich_df.to_csv(OUTPUT_DIR / "search_report.csv", index=False)
    # Convert DF to JSON safely (handle lists in cells)
    rich_df.to_json(
        OUTPUT_DIR / "search_report.json", orient="records", indent=2, force_ascii=False
    )

    logger.info(f"Found [green]{len(rich_df)}[/green] matching topics with metadata")
    return rich_df


if __name__ == "__main__":
    from mocks import DOCS

    run_topic_docs_search_demo(DOCS, "artificial intelligence")

    # --- Final Summary ---
    table = Table(title="Generated Artifacts", header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Description")

    artifacts = [
        ("inputs.json", "Source documents and query"),
        ("config.json", "Search & representation limits"),
        ("search_report.csv", "Tabular results with keywords & docs"),
        ("search_report.json", "Machine-readable enriched results"),
    ]

    for fname, desc in artifacts:
        link = f"file://{OUTPUT_DIR / fname}"
        table.add_row(f"[link={link}]{fname}[/link]", desc)

    console.print()
    console.print(
        Panel(
            table,
            title=f"[bold]Results: {Path(__file__).stem}[/bold]",
            border_style="blue",
        )
    )
