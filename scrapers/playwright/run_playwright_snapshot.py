#!/usr/bin/env python3
"""
Run Playwright ARIA Snapshot Generator
CLI tool to capture ARIA snapshots from URLs or local HTML files using Playwright.
"""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.scrapers.browser.playwright.snapshot import (
    capture_snapshot,
)
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture ARIA snapshots from URLs or local HTML files using Playwright",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture snapshot from URL
  python run_playwright_snapshot.py https://www.python.org
  
  # Capture snapshot from local HTML file
  python run_playwright_snapshot.py ./page.html
  
  # Capture interactive elements only
  python run_playwright_snapshot.py https://news.ycombinator.com --filter interactives
  
  # Use default mode instead of AI mode
  python run_playwright_snapshot.py https://docs.python.org --mode default
  
  # Show bounding boxes
  python run_playwright_snapshot.py https://github.com --boxes
  
  # Specify custom output directory
  python run_playwright_snapshot.py https://stackoverflow.com --output-dir ./my_snapshots
        """,
    )

    parser.add_argument(
        "source",
        type=str,
        help="URL or path to local HTML file",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30000,
        help="Maximum time in milliseconds to wait for page load (default: 30000)",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Maximum depth of the accessibility tree to capture (default: None/unlimited)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["ai", "default"],
        default="ai",
        help="Snapshot mode: 'ai' for AI-optimized or 'default' (default: ai)",
    )

    parser.add_argument(
        "--boxes",
        action="store_true",
        help="Include bounding box information in the snapshot",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (default: False/shows browser)",
    )

    parser.add_argument(
        "--filter",
        type=str,
        choices=["all", "interactives", "non-interactives", "links", "rag", "fields"],
        default="all",
        help="Filter type for snapshot content (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Directory to save snapshot outputs (default: {OUTPUT_DIR})",
    )

    return parser.parse_args()


async def main():
    """Main entry point for the snapshot capture tool."""
    args = parse_args()

    # Validate source
    source_path = Path(args.source)
    if not source_path.is_file() and not args.source.startswith(
        ("http://", "https://")
    ):
        console.print(
            f"[bold red]Error:[/bold red] Source must be a valid URL or existing file path"
        )
        console.print(f"Provided: {args.source}")
        sys.exit(1)

    if source_path.is_file():
        console.print(f"[dim]Source type: Local file[/dim]")
        console.print(f"[dim]File path: {source_path.absolute()}[/dim]")
    else:
        console.print(f"[dim]Source type: URL[/dim]")
        console.print(f"[dim]URL: {args.source}[/dim]")

    console.print("\n[bold blue]Configuration:[/bold blue]")
    console.print(f"  Timeout: {args.timeout}ms")
    console.print(f"  Depth: {args.depth if args.depth is not None else 'unlimited'}")
    console.print(f"  Mode: {args.mode}")
    console.print(f"  Boxes: {args.boxes}")
    console.print(f"  Headless: {args.headless}")
    console.print(f"  Filter: {args.filter}")
    console.print(f"  Output Dir: {args.output_dir}")
    console.print()

    # Handle output directory
    output_dir_path = Path(args.output_dir)
    if output_dir_path.exists():
        shutil.rmtree(output_dir_path)
        console.print(
            f"[bold yellow]Resetting output directory:[/bold yellow] {output_dir_path}"
        )
        console.print(f"[dim]Deleted existing directory[/dim]")

    output_dir_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Output directory ready:[/green] {output_dir_path}")

    try:
        # Capture the snapshot
        snapshot = await capture_snapshot(
            source=args.source,
            timeout=args.timeout,
            depth=args.depth,
            mode=args.mode,
            boxes=args.boxes,
            headless=args.headless,
            filter_type=args.filter,
        )

        # Generate default filename based on source
        if source_path.is_file():
            base_name = source_path.stem
        else:
            # Extract domain from URL
            from urllib.parse import urlparse

            parsed = urlparse(args.source)
            base_name = parsed.netloc.replace(".", "_")

        output_path = output_dir_path / f"{base_name}_snapshot.yaml"

        # Save the snapshot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(snapshot)
        console.print(
            f"\n[green]✓ Snapshot saved to:[/green] [link=file://{output_path.absolute()}]{output_path.stem}[/link]"
        )

        snapshot_tokens = count_tokens(snapshot, model=LLM_MODEL)
        console.print(f"[bold green]Tokens: {snapshot_tokens}[/bold green]")

        return 0

    except Exception as e:
        console.print(f"\n[bold red]Error capturing snapshot:[/bold red] {e}")
        console.print_exception()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
