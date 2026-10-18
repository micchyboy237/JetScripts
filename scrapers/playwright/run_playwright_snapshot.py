import argparse
import asyncio
import shutil
from pathlib import Path
from typing import Literal

from playwright.async_api import async_playwright
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

DEFAULT_URL = "https://ph.talent.com"


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture ARIA snapshot of a webpage using Playwright."
    )
    parser.add_argument(
        "url",
        type=str,
        nargs="?",
        default=DEFAULT_URL,
        help=f"URL of the webpage to capture snapshot from (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to save the snapshot (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="snapshot.md",
        help="Filename for the snapshot (default: snapshot.md)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=30000,
        help="Maximum time in milliseconds to wait for snapshot (default: 30000)",
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


async def capture_snapshot(
    url: str,
    timeout: float = 30000,
    depth: int | None = None,
    mode: Literal["ai", "default"] = "ai",
    boxes: bool = False,
    headless: bool = False,
) -> str:
    """
    Capture ARIA snapshot of a webpage.

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

    # Always print logs - verbose flag removed
    console.print(f"[bold blue]Starting snapshot capture for:[/bold blue] {url}")
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
        # Capture snapshot
        snapshot = asyncio.run(
            capture_snapshot(
                url=args.url,
                timeout=args.timeout,
                depth=args.depth,
                mode=args.mode,
                boxes=args.boxes,
                headless=args.headless,
            )
        )

        # Save snapshot to file
        snapshot_path = args.output_dir / args.filename
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(snapshot)

        # Display saved path as resource link
        console.print()
        console.print(
            f"[green]📄 Snapshot saved:[/green] "
            f"[link=file://{snapshot_path.resolve()}]{snapshot_path.name}[/link]"
        )

    except Exception as e:
        console.print(f"[bold red]Failed to capture snapshot:[/bold red] {e}")
        raise


if __name__ == "__main__":
    main()
