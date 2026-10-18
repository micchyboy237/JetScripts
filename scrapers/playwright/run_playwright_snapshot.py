"""
Playwright ARIA Snapshot Generator

Captures ARIA snapshots from live URLs or local HTML files using Playwright.
This tool is useful for generating accessibility trees for AI analysis, testing,
or documentation purposes. It supports both remote webpages and local HTML strings,
allowing for flexible snapshot generation in headless or headed modes.

Usage Examples:
    # Capture snapshot from a single URL
    python run_playwright_snapshot.py https://ph.talent.com

    # Capture snapshot from a local HTML file
    python run_playwright_snapshot.py data/__sample.html

    # Capture snapshots from multiple sources (URLs and files)
    python run_playwright_snapshot.py https://ph.talent.com data/__sample.html

    # Custom output directory and filename
    python run_playwright_snapshot.py https://example.com -o ./output -f my_snapshot.md

    # Use default mode and include bounding boxes
    python run_playwright_snapshot.py https://example.com -m default -b

    # Run in headed mode (visible browser) with custom timeout
    python run_playwright_snapshot.py https://example.com -H -t 60000
"""

import argparse
import asyncio
import shutil
from pathlib import Path
from typing import Literal

from playwright.async_api import async_playwright
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture ARIA snapshots of webpages or HTML files using Playwright."
    )
    parser.add_argument(
        "url_or_file",
        type=str,
        nargs="+",
        help="One or more URLs or paths to HTML files to capture snapshots from.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to save the snapshots (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Base filename for the snapshots. If multiple inputs are provided, an index will be appended (e.g., snapshot_0.md). Default: 'snapshot.md' or auto-generated based on input.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=30000,
        help="Maximum time in milliseconds to wait for each snapshot (default: 30000)",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=None,
        help="Limit the depth of the snapshot hierarchy (default: None/unlimited)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["ai", "default"],
        default="ai",
        help="Snapshot mode: 'ai' for AI-optimized, 'default' for standard (default: ai)",
    )
    parser.add_argument(
        "-b",
        "--boxes",
        action="store_true",
        default=False,
        help="Include bounding box coordinates in the snapshot (default: False)",
    )
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (default: False)",
    )
    return parser.parse_args()


async def capture_snapshot_from_url(
    url: str,
    timeout: float = 30000,
    depth: int | None = None,
    mode: Literal["ai", "default"] = "ai",
    boxes: bool = False,
    headless: bool = False,
) -> str:
    """
    Capture ARIA snapshot of a webpage from a URL.

    Args:
        url: URL of the webpage to capture snapshot from.
        timeout: Maximum time in milliseconds to wait for snapshot.
        depth: Limit the depth of the snapshot hierarchy.
        mode: Snapshot mode ('ai' or 'default').
        boxes: Include bounding box coordinates in the snapshot.
        headless: Run browser in headless mode.

    Returns:
        The ARIA snapshot as a string.
    """
    console.print(f"[bold blue]Starting snapshot capture for URL:[/bold blue] {url}")
    console.print(
        f"[dim]Parameters: timeout={timeout}ms, depth={depth}, mode={mode}, boxes={boxes}, headless={headless}[/dim]"
    )

    async with async_playwright() as p:
        console.print("[bold blue]Launching browser...[/bold blue]")
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        try:
            # Navigate to the page
            console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
            await page.goto(url, timeout=timeout)
            console.print("[green]Page loaded successfully[/green]")

            # Wait for page to stabilize (optional, helps with dynamic content)
            await page.wait_for_load_state("networkidle", timeout=timeout)
            console.print("[dim]Page network idle state reached[/dim]")

            # Capture ARIA snapshot
            console.print("[bold blue]Capturing ARIA snapshot...[/bold blue]")
            snapshot = await page.aria_snapshot(
                timeout=timeout,
                depth=depth,
                mode=mode,
                boxes=boxes,
            )
            console.print("[green]ARIA snapshot captured successfully[/green]")
            console.print(f"[dim]Snapshot size: {len(snapshot)} characters[/dim]")

            return snapshot

        except Exception as e:
            console.print(f"[bold red]Error during snapshot capture:[/bold red] {e}")
            raise
        finally:
            console.print("[bold blue]Closing browser...[/bold blue]")
            await browser.close()
            console.print("[green]Browser closed[/green]")


async def capture_snapshot_from_html(
    html_string: str,
    timeout: float = 30000,
    depth: int | None = None,
    mode: Literal["ai", "default"] = "ai",
    boxes: bool = False,
    headless: bool = False,
) -> str:
    """
    Capture ARIA snapshot from an HTML string.

    Args:
        html_string: The HTML content to load.
        timeout: Maximum time in milliseconds to wait for snapshot.
        depth: Limit the depth of the snapshot hierarchy.
        mode: Snapshot mode ('ai' or 'default').
        boxes: Include bounding box coordinates in the snapshot.
        headless: Run browser in headless mode.

    Returns:
        The ARIA snapshot as a string.
    """
    console.print("[bold blue]Starting snapshot capture from HTML string[/bold blue]")
    console.print(
        f"[dim]Parameters: timeout={timeout}ms, depth={depth}, mode={mode}, boxes={boxes}, headless={headless}[/dim]"
    )

    async with async_playwright() as p:
        console.print("[bold blue]Launching browser...[/bold blue]")
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        try:
            # Load the HTML string into the page
            console.print("[bold blue]Loading HTML content...[/bold blue]")
            await page.set_content(html_string)
            console.print("[green]HTML content loaded successfully[/green]")

            # Wait for any potential JS to execute
            await page.wait_for_load_state("networkidle", timeout=timeout)
            console.print("[dim]Page network idle state reached[/dim]")

            # Capture ARIA snapshot
            console.print("[bold blue]Capturing ARIA snapshot...[/bold blue]")
            snapshot = await page.aria_snapshot(
                timeout=timeout,
                depth=depth,
                mode=mode,
                boxes=boxes,
            )
            console.print("[green]ARIA snapshot captured successfully[/green]")
            console.print(f"[dim]Snapshot size: {len(snapshot)} characters[/dim]")

            return snapshot

        except Exception as e:
            console.print(f"[bold red]Error during snapshot capture:[/bold red] {e}")
            raise
        finally:
            console.print("[bold blue]Closing browser...[/bold blue]")
            await browser.close()
            console.print("[green]Browser closed[/green]")


def main():
    """Main entry point."""
    args = get_args()

    # Clean and recreate output directory
    console.print(
        f"[bold blue]Cleaning output directory:[/bold blue] {args.output_dir}"
    )
    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for index, input_source in enumerate(args.url_or_file):
            console.print(
                f"\n[bold cyan]Processing input {index + 1}/{len(args.url_or_file)}:[/bold cyan] {input_source}"
            )

            # Determine if input is a file path or URL
            input_path = Path(input_source)
            if input_path.is_file():
                # It's a file path
                console.print(
                    f"[bold blue]Reading HTML from file:[/bold blue] {input_path}"
                )
                with open(input_path, "r", encoding="utf-8") as file:
                    html = file.read()
                snapshot = asyncio.run(
                    capture_snapshot_from_html(
                        html_string=html,
                        timeout=args.timeout,
                        depth=args.depth,
                        mode=args.mode,
                        boxes=args.boxes,
                        headless=args.headless,
                    )
                )
                # Determine filename
                if args.filename:
                    base_name = Path(args.filename).stem
                    ext = Path(args.filename).suffix or ".md"
                    if len(args.url_or_file) > 1:
                        final_filename = f"{base_name}_{index}{ext}"
                    else:
                        final_filename = f"{base_name}{ext}"
                else:
                    if len(args.url_or_file) > 1:
                        final_filename = f"snapshot_{index}.md"
                    else:
                        final_filename = f"{input_path.stem}_snapshot.md"
            else:
                # Assume it's a URL
                snapshot = asyncio.run(
                    capture_snapshot_from_url(
                        url=input_source,
                        timeout=args.timeout,
                        depth=args.depth,
                        mode=args.mode,
                        boxes=args.boxes,
                        headless=args.headless,
                    )
                )
                # Determine filename
                if args.filename:
                    base_name = Path(args.filename).stem
                    ext = Path(args.filename).suffix or ".md"
                    if len(args.url_or_file) > 1:
                        final_filename = f"{base_name}_{index}{ext}"
                    else:
                        final_filename = f"{base_name}{ext}"
                else:
                    # Create a safe filename from URL
                    safe_url = (
                        input_source.replace("://", "_")
                        .replace("/", "_")
                        .replace(":", "_")
                    )
                    if len(args.url_or_file) > 1:
                        final_filename = f"{safe_url[:50]}_{index}.md"
                    else:
                        final_filename = f"{safe_url[:50]}.md"

            # Save snapshot to file
            snapshot_path = args.output_dir / final_filename
            with open(snapshot_path, "w", encoding="utf-8") as f:
                f.write(snapshot)

            # Display saved path as resource link
            console.print(
                f"[green]📄 Snapshot saved:[/green] "
                f"[link=file://{snapshot_path.resolve()}]{snapshot_path.name}[/link]"
            )

    except Exception as e:
        console.print(f"[bold red]Failed to capture snapshot:[/bold red] {e}")
        raise


if __name__ == "__main__":
    main()
