"""
01_demo_topic_extraction.py
Extract topics from text documents using BERTopic with local embeddings.
Saves config, inputs, and results to OUTPUT_DIR.
"""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from jet.libs.bertopic.monkey_patches.add_check_array import init_patch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

init_patch()

from typing import List

from jet.adapters.bertopic.factory import (
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
    top_n_words: int = 5,
) -> TopicExtractionResult:
    """
    Demonstrate topic extraction using the reusable factory functions.
    """
    # 1. Save Inputs
    input_path = OUTPUT_DIR / "inputs.json"
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump({"documents": documents}, f, indent=2, ensure_ascii=False)
    logger.info(
        f"Saved [cyan]{len(documents)}[/cyan] input documents to {input_path.name}"
    )

    # 2. Save Config
    config = {
        "min_topic_size": min_topic_size,
        "top_n_words": top_n_words,
        "embedder": "llama_cpp_local",
    }
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {config_path.name}")

    # 3. Run Extraction
    embedder = create_bertopic_embedder()
    sanity_check_embedder(embedder)

    logger.info("Starting topic extraction...")
    result = extract_topics(
        documents=documents,
        embedder=embedder,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        verbose=False,  # We handle logging via rich
    )

    # 4. Save Outputs
    # Save serializable topic data
    topics_out_path = OUTPUT_DIR / "topics.json"
    with open(topics_out_path, "w", encoding="utf-8") as f:
        json.dump(result["topics"], f, indent=2, ensure_ascii=False)

    # Save embeddings as binary
    emb_path = OUTPUT_DIR / "embeddings.npy"
    np.save(emb_path, result["embeddings"])

    # Save topic info dataframe
    info_path = OUTPUT_DIR / "topic_info.csv"
    result["topic_info"].to_csv(info_path, index=False)

    logger.info(f"Extracted [green]{len(result['topics'])}[/green] topics")
    return result


if __name__ == "__main__":
    from mocks import DOCS

    sample_docs = DOCS
    result = run_topic_extraction_demo(sample_docs)

    # --- Final Summary ---
    table = Table(
        title="Generated Artifacts", show_header=True, header_style="bold magenta"
    )
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Description")

    artifacts = [
        ("inputs.json", "Input", f"{len(sample_docs)} source documents"),
        ("config.json", "Config", "Extraction parameters"),
        ("topics.json", "Output", "Structured topic keywords & docs"),
        (
            "embeddings.npy",
            "Output",
            f"Document embeddings {result['embeddings'].shape}",
        ),
        ("topic_info.csv", "Output", "BERTopic summary statistics"),
    ]

    for fname, ftype, desc in artifacts:
        link = f"file://{OUTPUT_DIR / fname}"
        table.add_row(f"[link={link}]{fname}[/link]", ftype, desc)

    console.print()
    console.print(
        Panel(
            table,
            title=f"[bold]Results: {Path(__file__).stem}[/bold]",
            border_style="blue",
        )
    )
