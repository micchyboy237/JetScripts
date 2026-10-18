import argparse
import asyncio
import shutil
from pathlib import Path

from playwright.async_api import async_playwright
from rich.console import Console

console = Console()

# Configuration
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DEFAULT_URL = "https://duckduckgo.com"
DEFAULT_TEXT = "Playwright MCP Demo"


def get_args():
    parser = argparse.ArgumentParser(
        description="Type text into a search bar using Playwright."
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"URL to visit (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "-t", "--text", type=str, default=DEFAULT_TEXT, help="Text to type"
    )
    parser.add_argument(
        "-s", "--slowly", action="store_true", help="Type character by character"
    )
    parser.add_argument(
        "-S", "--submit", action="store_true", help="Press Enter after typing"
    )
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode",
    )
    return parser.parse_args()


async def type_text_demo(
    url: str,
    text: str,
    slowly: bool = False,
    submit: bool = False,
    headless: bool = False,
):
    console.print(f"[bold blue]Starting type demo for:[/bold blue] {url}")
    console.print(f"[dim]Text: '{text}', Slowly: {slowly}, Submit: {submit}[/dim]")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        try:
            console.print("[bold blue]Launching browser...[/bold blue]")
            await asyncio.sleep(1)  # Delay to see browser open

            console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
            # Use wait_until='networkidle' to ensure JS-heavy sites like DDG are fully ready
            await page.goto(url, wait_until="networkidle")
            await asyncio.sleep(2)  # Delay to see the page load

            # Use a more resilient selector.
            locator = page.get_by_placeholder("Search privately")

            # Fallback to role-based if placeholder fails
            if await locator.count() == 0:
                locator = page.get_by_role("searchbox")

            console.print(
                "[bold blue]Waiting for search box to be visible...[/bold blue]"
            )
            await locator.wait_for(state="visible", timeout=10000)
            await asyncio.sleep(1)  # Delay to see focus/wait

            console.print("[bold blue]Typing text...[/bold blue]")
            if slowly:
                await locator.press_sequentially(
                    text, delay=100
                )  # Add delay between chars
            else:
                await locator.fill(text)

            await asyncio.sleep(2)  # Delay to see the text appear

            if submit:
                console.print("[bold blue]Submitting search...[/bold blue]")
                await locator.press("Enter")
                await page.wait_for_load_state("networkidle")
                await asyncio.sleep(2)  # Delay to see results load

            console.print("[green]Typing complete![/green]")

            # Save snapshot of state
            output_path = OUTPUT_DIR / "typed_state.html"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(await page.content())
            console.print(
                f"[green]State saved to [link=file://{output_path.resolve()}]{output_path.name}[/link][/green]"
            )

            await asyncio.sleep(
                3
            )  # Final delay before closing so you can see the result

        except Exception as e:
            console.print(f"[bold red]Error during typing:[/bold red] {e}")
            raise
        finally:
            console.print("[bold blue]Closing browser...[/bold blue]")
            await browser.close()


def main():
    args = get_args()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(
            type_text_demo(args.url, args.text, args.slowly, args.submit, args.headless)
        )
    except Exception as e:
        console.print(f"[bold red]Failed:[/bold red] {e}")


if __name__ == "__main__":
    main()
