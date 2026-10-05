"""
05_demo_hierarchy.py
Demonstrates explore_hierarchy for topic relationships.
Saves hierarchy merge table and tree structure.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List

from jet.libs.bertopic.monkey_patches.add_check_array import init_patch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

init_patch()

from jet.adapters.bertopic.factory_with_repr import (
    create_bertopic_embedder,
    create_topic_model,
    explore_hierarchy,
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


def run_hierarchy_demo(documents: List[str]):
    """Build and save topic hierarchy."""

    # Save Inputs
    with open(OUTPUT_DIR / "inputs.json", "w", encoding="utf-8") as f:
        json.dump({"documents": documents}, f, indent=2, ensure_ascii=False)

    config = {"linkage": "ward", "use_ctfidf": True}
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Fit Model
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)
    topic_model = create_topic_model(embedder=embedder, min_topic_size=2, verbose=False)
    embeddings = embedder.embed(documents, verbose=False)
    _, _ = topic_model.fit_transform(documents, embeddings=embeddings)

    # Explore Hierarchy
    logger.info("Building topic hierarchy (Ward linkage)...")
    hier_df = explore_hierarchy(topic_model, documents, linkage="ward", verbose=False)

    # Save Outputs
    hier_df.to_csv(OUTPUT_DIR / "hierarchy_merges.csv", index=False)

    # Save text-based tree preview
    try:
        tree_str = topic_model.get_topic_tree(hier_df)
        with open(OUTPUT_DIR / "topic_tree.txt", "w", encoding="utf-8") as f:
            f.write(tree_str)
    except Exception as e:
        logger.warning(f"Could not generate topic tree text: {e}")

    logger.info(f"Hierarchy built with [green]{len(hier_df)}[/green] merges")
    return hier_df


if __name__ == "__main__":
    from mocks import DOCS

    run_hierarchy_demo(DOCS)

    # --- Final Summary ---
    table = Table(title="Generated Artifacts", header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Description")

    artifacts = [
        ("inputs.json", "Source corpus for clustering"),
        ("config.json", "Linkage method and c-TF-IDF settings"),
        ("hierarchy_merges.csv", "Parent-child merge distances & IDs"),
        ("topic_tree.txt", "ASCII visualization of topic tree"),
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
