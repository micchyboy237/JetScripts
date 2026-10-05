"""
02_demo_topic_extraction_repr.py
Enhanced topic extraction with improved representation models.
Saves config, inputs, and enriched results to OUTPUT_DIR.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from jet.libs.bertopic.monkey_patches.add_check_array import init_patch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

init_patch()

from jet.adapters.bertopic.factory_with_repr import (
    TopicExtractionResult,
    create_bertopic_embedder,
    extract_topics,
    sanity_check_embedder,
)

# --- Output Directory Setup ---
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Rich Logging Setup ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("rich")
console = Console()


def run_topic_extraction_demo(
    documents: List[str],
    min_topic_size: int = 3,
    top_n_words: int = 10,
    n_representative_docs: Optional[int] = None,
) -> TopicExtractionResult:
    """Demonstrate enhanced topic extraction with KeyBERT + stopword removal."""

    # Save Inputs
    with open(OUTPUT_DIR / "inputs.json", "w", encoding="utf-8") as f:
        json.dump({"documents": documents}, f, indent=2, ensure_ascii=False)

    # Save Config
    config = {
        "min_topic_size": min_topic_size,
        "top_n_words": top_n_words,
        "n_representative_docs": n_representative_docs,
        "remove_stop_words": True,
        "use_keybert": True,
    }
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Run Extraction
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)

    logger.info("Running enhanced topic extraction (KeyBERT+StopWords)...")
    result = extract_topics(
        documents=documents,
        embedder=embedder,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        remove_stop_words=True,
        use_keybert=True,
        verbose=False,
        n_representative_docs=n_representative_docs,
    )

    # Save Outputs
    with open(OUTPUT_DIR / "topics_enriched.json", "w", encoding="utf-8") as f:
        json.dump(result["topics"], f, indent=2, ensure_ascii=False)

    np.save(OUTPUT_DIR / "embeddings.npy", result["embeddings"])
    result["topic_info"].to_csv(OUTPUT_DIR / "topic_info.csv", index=False)

    logger.info(f"Extracted [green]{len(result['topics'])}[/green] enhanced topics")
    return result


if __name__ == "__main__":
    from mocks import DOCS

    result = run_topic_extraction_demo(DOCS)

    # --- Final Summary ---
    table = Table(title="Generated Artifacts", header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Description")

    artifacts = [
        ("inputs.json", f"{len(DOCS)} source documents"),
        ("config.json", "KeyBERT & StopWord configuration"),
        ("topics_enriched.json", "Topics with refined keywords & rep docs"),
        ("embeddings.npy", f"Embeddings array {result['embeddings'].shape}"),
        ("topic_info.csv", "Topic statistics & representations"),
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
