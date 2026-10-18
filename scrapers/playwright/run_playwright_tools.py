import asyncio
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from rich.console import Console

console = Console()

# Configuration
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_URL = "https://ph.talent.com"
DEFAULT_TIMEOUT = 30000


class PlaywrightToolbox:
    def __init__(self, headless: bool = False, output_dir: str | Path = OUTPUT_DIR):
        self.headless = headless
        self.output_dir = Path(output_dir)
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def start(self):
        """Initialize the browser and context."""
        console.print("[bold blue]Initializing Playwright...[/bold blue]")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        self.page = await self.context.new_page()
        console.print("[green]Browser ready[/green]")

    async def close(self):
        """Close the browser."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        console.print("[green]Browser closed[/green]")

    async def _ensure_page(self) -> Page:
        if not self.page or self.page.is_closed():
            self.page = await self.context.new_page()
        return self.page

    # --- Navigation Tools ---

    async def browser_navigate(self, url: str):
        """Navigate to a URL"""
        page = await self._ensure_page()
        console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
        await page.goto(url)
        console.print("[green]Navigation complete[/green]")

    async def browser_navigate_back(self):
        """Go back to the previous page"""
        page = await self._ensure_page()
        console.print("[bold blue]Going back[/bold blue]")
        await page.go_back()
        console.print("[green]Back navigation complete[/green]")

    # --- Interaction Tools ---

    async def browser_type(
        self,
        target: str,
        text: str,
        element: Optional[str] = None,
        submit: Optional[bool] = False,
        slowly: Optional[bool] = False,
    ):
        """Type text into editable element"""
        page = await self._ensure_page()
        console.print(f"[bold blue]Typing '{text}' into target:[/bold blue] {target}")
        locator = page.locator(target)
        if slowly:
            await locator.press_sequentially(text)
        else:
            await locator.fill(text)

        if submit:
            await locator.press("Enter")
        console.print("[green]Typing complete[/green]")

    async def browser_press_key(self, key: str):
        """Press a key on the keyboard"""
        page = await self._ensure_page()
        console.print(f"[bold blue]Pressing key:[/bold blue] {key}")
        await page.keyboard.press(key)
        console.print("[green]Key press complete[/green]")

    async def browser_hover(self, target: str, element: Optional[str] = None):
        """Hover over element on page"""
        page = await self._ensure_page()
        console.print(f"[bold blue]Hovering over target:[/bold blue] {target}")
        locator = page.locator(target)
        await locator.hover()
        console.print("[green]Hover complete[/green]")

    async def browser_click(
        self,
        target: str,
        element: Optional[str] = None,
        doubleClick: Optional[bool] = False,
        button: Optional[Literal["left", "right", "middle"]] = "left",
        modifiers: Optional[List[str]] = None,
    ):
        """Perform click on a web page"""
        page = await self._ensure_page()
        console.print(f"[bold blue]Clicking target:[/bold blue] {target}")
        locator = page.locator(target)

        options = {"button": button}
        if modifiers:
            options["modifiers"] = modifiers

        if doubleClick:
            await locator.dblclick(**options)
        else:
            await locator.click(**options)
        console.print("[green]Click complete[/green]")

    async def browser_fill_form(self, fields: List[dict]):
        """Fill multiple form fields"""
        page = await self._ensure_page()
        console.print(f"[bold blue]Filling form with {len(fields)} fields[/bold blue]")
        for field in fields:
            name = field.get("name")
            target = field.get("target")
            value = field.get("value")
            type_ = field.get("type")

            locator = page.locator(target)
            if type_ == "textbox":
                await locator.fill(value)
            elif type_ == "checkbox":
                if value.lower() == "true":
                    await locator.check()
                else:
                    await locator.uncheck()
            elif type_ == "combobox":
                await locator.select_option(value)
        console.print("[green]Form fill complete[/green]")

    async def browser_select_option(
        self, target: str, values: List[str], element: Optional[str] = None
    ):
        """Select an option in a dropdown"""
        page = await self._ensure_page()
        console.print(
            f"[bold blue]Selecting options {values} in target:[/bold blue] {target}"
        )
        locator = page.locator(target)
        await locator.select_option(values)
        console.print("[green]Option selection complete[/green]")

    async def browser_handle_dialog(
        self, accept: bool, promptText: Optional[str] = None
    ):
        """Handle a dialog"""
        page = await self._ensure_page()
        console.print(f"[bold blue]Handling dialog (accept={accept})[/bold blue]")
        # Note: In a real scenario, we'd wait for the dialog event.
        # This is a simplified representation for the tool definition.
        async with page.expect_dialog() as dialog_info:
            # Trigger action that causes dialog would happen here in a real flow
            pass
        dialog = dialog_info.value
        if accept:
            await dialog.accept(promptText)
        else:
            await dialog.dismiss()
        console.print("[green]Dialog handled[/green]")

    # --- Inspection & Evaluation Tools ---

    async def browser_find(
        self,
        text: Optional[str] = None,
        regex: Optional[str] = None,
        max_results: Optional[int] = None,
        filename: Optional[str] = None,
    ):
        """Find in page snapshot"""
        page = await self._ensure_page()
        console.print(f"[bold blue]Finding text:[/bold blue] {text or regex}")
        # Simplified implementation using page content
        content = await page.content()
        results = []
        if text:
            if text.lower() in content.lower():
                results.append(f"Found '{text}'")
        if regex:
            import re

            matches = re.findall(regex, content)
            if matches:
                results.extend(matches[:max_results])

        if self.output_dir:
            output_path = self.output_dir / (filename or "find_results.txt")
            with open(output_path, "w") as f:
                f.write("\n".join(results))
            console.print(
                f"[green]Results saved to [link=file://{output_path.resolve()}]{output_path.name}[/link][/green]"
            )

    async def browser_evaluate(
        self,
        function: str,
        element: Optional[str] = None,
        target: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        """Evaluate JavaScript expression on page or element"""
        page = await self._ensure_page()
        console.print(f"[bold blue]Evaluating function:[/bold blue] {function[:50]}...")
        if target:
            locator = page.locator(target)
            result = await locator.evaluate(function)
        else:
            result = await page.evaluate(function)

        output_path = self.output_dir / (filename or "eval_result.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        console.print(
            f"[green]Result saved to [link=file://{output_path.resolve()}]{output_path.name}[/link][/green]"
        )
        return result

    async def browser_run_code_unsafe(
        self, code: Optional[str] = None, filename: Optional[str] = None
    ):
        """Run Playwright code (unsafe)"""
        page = await self._ensure_page()
        console.print(
            f"[bold red]Running unsafe code:[/bold red] {code[:50] if code else filename}..."
        )
        if filename:
            with open(self.output_dir / filename, "r") as f:
                code = f.read()

        result = await page.evaluate(code)
        console.print(f"[green]Code execution complete[/green]")
        return result

    async def browser_snapshot(
        self,
        target: Optional[str] = None,
        filename: Optional[str] = None,
        depth: Optional[int] = None,
        boxes: Optional[bool] = False,
    ):
        """Capture accessibility snapshot of the current page"""
        page = await self._ensure_page()
        console.print("[bold blue]Capturing snapshot[/bold blue]")
        snapshot = await page.aria_snapshot(depth=depth, boxes=boxes)

        output_path = self.output_dir / (filename or "snapshot.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(snapshot)
        console.print(
            f"[green]Snapshot saved to [link=file://{output_path.resolve()}]{output_path.name}[/link][/green]"
        )
        return snapshot

    async def browser_take_screenshot(
        self,
        element: Optional[str] = None,
        target: Optional[str] = None,
        type: Optional[Literal["png", "jpeg", "webp"]] = "png",
        filename: Optional[str] = None,
        fullPage: Optional[bool] = False,
        scale: Optional[Literal["css", "device"]] = "css",
    ):
        """Take a screenshot of the current page"""
        page = await self._ensure_page()
        console.print(
            f"[bold blue]Taking screenshot (type={type}, fullPage={fullPage})[/bold blue]"
        )

        options = {"type": type, "scale": scale, "full_page": fullPage}

        if type == "jpeg":
            options["quality"] = 90

        if target:
            locator = page.locator(target)
            data = await locator.screenshot(**options)
        else:
            data = await page.screenshot(**options)

        output_path = self.output_dir / (filename or f"screenshot.{type}")
        with open(output_path, "wb") as f:
            f.write(data)
        console.print(
            f"[green]Screenshot saved to [link=file://{output_path.resolve()}]{output_path.name}[/link][/green]"
        )
        return data

    async def browser_wait_for(
        self,
        time: Optional[float] = None,
        text: Optional[str] = None,
        textGone: Optional[str] = None,
    ):
        """Wait for text to appear or disappear or a specified time to pass"""
        page = await self._ensure_page()
        if time:
            console.print(f"[bold blue]Waiting for {time} seconds[/bold blue]")
            await page.wait_for_timeout(time * 1000)
        elif text:
            console.print(f"[bold blue]Waiting for text:[/bold blue] {text}")
            await page.get_by_text(text).first.wait_for(state="visible")
        elif textGone:
            console.print(
                f"[bold blue]Waiting for text to disappear:[/bold blue] {textGone}"
            )
            await page.get_by_text(textGone).first.wait_for(state="hidden")
        console.print("[green]Wait complete[/green]")

    async def browser_close(self):
        """Close the page"""
        console.print("[bold blue]Closing browser[/bold blue]")
        await self.close()
        console.print("[green]Browser closed[/green]")


async def main():
    """Example usage"""
    toolbox = PlaywrightToolbox(headless=False, output_dir=OUTPUT_DIR)

    try:
        await toolbox.start()
        await toolbox.browser_navigate(DEFAULT_URL)
        await toolbox.browser_wait_for(time=2)
        await toolbox.browser_snapshot(filename="initial_snapshot.md")
        await toolbox.browser_take_screenshot(fullPage=True)
    finally:
        await toolbox.close()


if __name__ == "__main__":
    asyncio.run(main())
