"""
03_demo_topic_search.py
Lightweight semantic search over fitted BERTopic topic centroids.
Saves query config, search results, and model metadata.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List

import numpy as np
from jet.libs.bertopic.monkey_patches.add_check_array import init_patch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

init_patch()

from jet.adapters.bertopic.factory_with_repr import (
    create_bertopic_embedder,
    create_topic_model,
    find_topics,
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


def run_topic_search_demo(documents: List[str], query: str, top_n: int = 5):
    """Fit topic model and search for relevant topics."""

    # Save Inputs & Config
    with open(OUTPUT_DIR / "inputs.json", "w", encoding="utf-8") as f:
        json.dump(
            {"documents": documents, "query": query}, f, indent=2, ensure_ascii=False
        )

    config = {"top_n": top_n, "min_topic_size": 2}
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Fit Model
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)
    topic_model = create_topic_model(embedder=embedder, min_topic_size=2, verbose=False)

    logger.info("Fitting topic model for search index...")
    embeddings = embedder.embed(documents, verbose=False)
    _, _ = topic_model.fit_transform(documents, embeddings=embeddings)

    # Search
    logger.info(f"Searching for: [yellow]'{query}'[/yellow]")
    similar_topics, similarities = find_topics(
        topic_model, query, top_n=top_n, verbose=False
    )

    # Save Results
    results = [
        {"rank": i + 1, "topic_id": int(tid), "similarity": float(sim)}
        for i, (tid, sim) in enumerate(zip(similar_topics, similarities))
    ]
    with open(OUTPUT_DIR / "search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    return similar_topics, similarities


if __name__ == "__main__":
    from mocks import DOCS

    run_topic_search_demo(DOCS, "artificial intelligence")

    # --- Final Summary ---
    table = Table(title="Generated Artifacts", header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Description")

    artifacts = [
        ("inputs.json", "Documents and search query"),
        ("config.json", "Search parameters"),
        ("search_results.json", "Ranked topic IDs and similarity scores"),
        ("embeddings.npy", "Index embeddings used for fitting"),
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
