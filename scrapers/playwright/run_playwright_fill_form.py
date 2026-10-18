import argparse
import asyncio
import shutil
from pathlib import Path

from playwright.async_api import async_playwright
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DEFAULT_URL = "https://httpbin.org/forms/post"


def get_args():
    parser = argparse.ArgumentParser(
        description="Fill out a web form using Playwright."
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"URL of the form (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode",
    )
    return parser.parse_args()


async def fill_form_demo(url: str = DEFAULT_URL, headless: bool = False):
    console.print(f"[bold blue]Starting form fill demo for:[/bold blue] {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        try:
            console.print("[bold blue]Navigating...[/bold blue]")
            # Wait for networkidle to ensure the simple httpbin form is fully ready
            await page.goto(url, wait_until="networkidle")

            console.print("[bold blue]Filling fields...[/bold blue]")

            # 1. Use get_by_label for the most robust selection based on the <label> text
            await page.get_by_label("Customer name").fill("John Doe")
            console.print("[green]Filled 'John Doe' into Customer name[/green]")

            await page.get_by_label("Telephone").fill("555-1234")
            console.print("[green]Filled '555-1234' into Telephone[/green]")

            await page.get_by_label("E-mail address").fill("john@example.com")
            console.print("[green]Filled 'john@example.com' into E-mail[/green]")

            # 2. For radios, use get_by_role with name
            await page.get_by_role("radio", name="Medium").check()
            console.print("[green]Selected 'Medium' for Pizza Size[/green]")

            # 3. For checkboxes, use get_by_role
            await page.get_by_role("checkbox", name="Bacon").check()
            await page.get_by_role("checkbox", name="Extra Cheese").check()
            console.print("[green]Checked Bacon and Extra Cheese[/green]")

            # 4. For time inputs, get_by_label works if the label is associated correctly
            await page.get_by_label("Preferred delivery time").fill("13:00")
            console.print("[green]Filled '13:00' for delivery time[/green]")

            # 5. For textarea
            await page.get_by_label("Delivery instructions").fill(
                "Please leave at the door."
            )
            console.print("[green]Filled delivery instructions[/green]")

            console.print("[bold blue]Submitting form...[/bold blue]")
            await page.get_by_role("button", name="Submit order").click()

            # Wait for response
            await page.wait_for_load_state("networkidle")
            console.print("[green]Form submitted successfully![/green]")

            # Save result
            output_path = OUTPUT_DIR / "form_result.html"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(await page.content())
            console.print(
                f"[green]Result saved to [link=file://{output_path.resolve()}]{output_path.name}[/link][/green]"
            )

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise
        finally:
            await browser.close()


def main():
    args = get_args()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(fill_form_demo(args.url, args.headless))


if __name__ == "__main__":
    main()
