# Complete the shell script that will create the file structure with full code given this discussion
# cd <path_to_base_dir>

#!/bin/bash

# Shell script to generate the example adaptive crawler Python files with full code.

# Set script to exit on error
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXAMPLES_DIR="$BASE_DIR/examples"
STATE_DIR="$EXAMPLES_DIR/state"
KNOWLEDGE_DIR="$EXAMPLES_DIR/knowledge"

mkdir -p "$EXAMPLES_DIR" "$STATE_DIR" "$KNOWLEDGE_DIR"

# Write examples/02_statistical_with_rich_output.py
cat > "$EXAMPLES_DIR/02_statistical_with_rich_output.py" << 'EOF'
"""
Example: Statistical adaptive crawler + rich console output

Shows how to get nice tables and colored output using the rich library.
"""

import asyncio
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.adaptive_crawler import AdaptiveCrawler, AdaptiveConfig


async def main():
    # ────────────────────────────────────────────────
    # Configuration ─ very conservative / fast settings
    # ────────────────────────────────────────────────
    config = AdaptiveConfig(
        strategy="statistical",
        max_pages=15,
        max_depth=4,
        confidence_threshold=0.75,
        top_k_links=3,
        min_gain_threshold=0.08,
        saturation_threshold=0.82,
        save_state=True,
        state_path=str(Path("state") / "statistical-example.json"),
    )

    # We'll let AdaptiveCrawler create its own crawler instance
    adaptive = AdaptiveCrawler(config=config)

    # Starting point & question
    start_url = "https://fastapi.tiangolo.com/tutorial/"
    query = "How do I add request validation and automatic docs in FastAPI?"

    print("\n" + "═" * 70)
    print(f" Starting STATISTICAL adaptive crawl")
    print(f" Query:   {query}")
    print(f" Start:   {start_url}")
    print("═" * 70 + "\n")

    # ── Run the adaptive crawl ────────────────────────────────
    state = await adaptive.digest(start_url=start_url, query=query)

    # ── Show results with rich formatting ─────────────────────
    print("\n" + "═" * 70)
    print(" Crawl finished")
    print("═" * 70)

    adaptive.print_stats(detailed=True)

    # Optional: also show the most relevant pieces of content
    print("\n" + "─" * 70)
    print("Top relevant content snippets (simple keyword overlap)")
    print("─" * 70)

    relevant = adaptive.get_relevant_content(top_k=4)
    from rich.console import Console
    console = Console()

    for i, item in enumerate(relevant, 1):
        console.print(f"[bold cyan]{i}.[/bold cyan] {item['url']}")
        console.print(f"   Score: [yellow]{item['score']:.3f}[/yellow]")
        preview = item['content'][:280].replace("\n", " ").strip()
        console.print(f"   [dim]{preview}…[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())
EOF

# Write examples/04_resume_from_state.py
cat > "$EXAMPLES_DIR/04_resume_from_state.py" << 'EOF'
"""
Example: Demonstrate resuming an interrupted / saved crawl

Shows how to:
- save state during crawl
- later resume from that state
- continue crawling until better confidence or max pages
"""

import asyncio
from pathlib import Path
import time

from crawl4ai import AsyncWebCrawler
from crawl4ai.adaptive_crawler import AdaptiveCrawler, AdaptiveConfig, CrawlState


STATE_PATH = Path("state") / "resume-example.json"


async def run_first_part():
    """First run — crawl a bit and save state"""
    config = AdaptiveConfig(
        strategy="statistical",
        max_pages=8,                # stop early on purpose
        max_depth=4,
        confidence_threshold=0.92,  # high → probably won't reach
        top_k_links=3,
        save_state=True,
        state_path=str(STATE_PATH),
    )

    adaptive = AdaptiveCrawler(config=config)

    start_url = "https://www.python.org/dev/peps/"
    query = "What is the current status of Python type hints and PEP 695?"

    print(f"\n[Part 1] Starting initial crawl (limited to {config.max_pages} pages)...")
    state = await adaptive.digest(start_url, query)

    print(f"\nSaved state to: {STATE_PATH}")
    print(f"Pages so far : {len(state.crawled_urls)}")
    print(f"Confidence    : {adaptive.confidence:.2%}")
    print(f"Stopped because: {state.metrics.get('stopped_reason', 'unknown')}")


async def run_resume():
    """Second run — resume from saved state and continue"""
    if not STATE_PATH.exists():
        print(f"State file not found: {STATE_PATH}")
        return

    print(f"\n[Part 2] Resuming crawl from saved state...")
    print(f"Loading: {STATE_PATH}")

    # We can create new config with different (usually more permissive) limits
    config = AdaptiveConfig(
        strategy="statistical",
        max_pages=18,               # allow more pages now
        confidence_threshold=0.82,
        top_k_links=4,
        save_state=True,
        state_path=str(STATE_PATH),  # will overwrite on further saves
    )

    adaptive = AdaptiveCrawler(config=config)

    # Load existing state and continue
    state = await adaptive.digest(
        start_url=None,              # not needed when resuming
        query=None,                  # will be read from state
        resume_from=str(STATE_PATH)
    )

    print("\nResume finished")
    adaptive.print_stats(detailed=False)


async def main():
    STATE_PATH.parent.mkdir(exist_ok=True, parents=True)

    # Run first part (limited crawl + save)
    await run_first_part()

    print("\n" + "─" * 60)
    print("You can now inspect/modify state or wait before resuming...")
    print("─" * 60 + "\n")

    # For demo we immediately resume — in real usage you might wait hours/days
    time.sleep(2)  # small pause so it's clear there are two phases

    await run_resume()


if __name__ == "__main__":
    asyncio.run(main())
EOF

# Write examples/05_export_import_knowledge_base.py
cat > "$EXAMPLES_DIR/05_export_import_knowledge_base.py" << 'EOF'
"""
Example: Export & import knowledge base (jsonl format)

Useful for:
- archiving crawl results
- using them later in RAG pipelines
- sharing between machines / sessions
"""

import asyncio
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.adaptive_crawler import AdaptiveCrawler, AdaptiveConfig


EXPORT_PATH = Path("knowledge") / "fastapi-tutorial.jsonl"


async def crawl_and_export():
    config = AdaptiveConfig(
        strategy="statistical",
        max_pages=10,
        max_depth=3,
        confidence_threshold=0.80,
        top_k_links=3,
        save_state=False,           # we only want the knowledge base
    )

    adaptive = AdaptiveCrawler(config=config)

    start_url = "https://fastapi.tiangolo.com/tutorial/"
    query = "How to handle form data and file uploads in FastAPI?"

    print(f"\nCrawling → {query}")
    state = await adaptive.digest(start_url, query)

    print(f"\nCrawl finished — {len(state.knowledge_base)} documents")

    # Export
    EXPORT_PATH.parent.mkdir(exist_ok=True, parents=True)
    adaptive.export_knowledge_base(EXPORT_PATH, format="jsonl")

    print(f"Exported to: {EXPORT_PATH}")
    print(f"Size: {EXPORT_PATH.stat().st_size:,} bytes")


async def import_and_show():
    if not EXPORT_PATH.exists():
        print(f"File not found: {EXPORT_PATH}")
        return

    print(f"\nImporting knowledge base from: {EXPORT_PATH}")

    # Create fresh crawler just for importing & viewing
    adaptive = AdaptiveCrawler()

    # Import documents into state
    adaptive.import_knowledge_base(EXPORT_PATH, format="jsonl")

    # Show stats (even though no real crawl happened this session)
    print("\nImported knowledge base statistics:")
    adaptive.print_stats(detailed=False)

    # Show a few sample documents
    print("\n" + "─" * 70)
    print("First 3 imported documents (url + preview)")
    print("─" * 70)

    from rich.console import Console
    console = Console()

    for i, doc in enumerate(adaptive.state.knowledge_base[:3], 1):
        preview = (doc.markdown.raw_markdown or "")[:220].replace("\n", " ").strip()
        console.print(f"[bold cyan]{i}.[/bold cyan] {doc.url}")
        console.print(f"   [dim]{preview}…[/dim]\n")


async def main():
    print("═" * 70)
    print(" Part 1 : Crawl + Export")
    print("═" * 70)
    await crawl_and_export()

    print("\n" + "═" * 70)
    print(" Part 2 : Import + Inspect")
    print("═" * 70)
    await import_and_show()


if __name__ == "__main__":
    asyncio.run(main())
EOF

echo "Example scripts have been created in: $EXAMPLES_DIR"
