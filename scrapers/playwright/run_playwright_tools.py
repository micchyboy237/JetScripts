import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import openai
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_URL = "https://ph.talent.com"
# Reduced timeouts for faster feedback during agent loops
ACTION_TIMEOUT = 5000
NAVIGATION_TIMEOUT = 10000


class PlaywrightToolbox:
    def __init__(self, headless: bool = False, output_dir: str | Path = OUTPUT_DIR):
        self.headless = headless
        self.output_dir = Path(output_dir)
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.last_snapshot: Optional[str] = None

    async def start(self):
        """Initialize the browser and context."""
        console.print("[bold blue]Initializing Playwright...[/bold blue]")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        self.page = await self.context.new_page()
        # Set global timeouts for the page
        self.page.set_default_timeout(ACTION_TIMEOUT)
        self.page.set_default_navigation_timeout(NAVIGATION_TIMEOUT)
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
            self.page.set_default_timeout(ACTION_TIMEOUT)
            self.page.set_default_navigation_timeout(NAVIGATION_TIMEOUT)
        return self.page

    def _resolve_target(self, target: str) -> str:
        """Resolve snapshot ref to a usable selector if possible."""
        # Check if target is a ref like e39
        if self.last_snapshot and re.match(r"^e\d+$", target):
            # Try to find the role, name, and testid associated with this ref in the snapshot
            pattern = rf"- (\w+) \"([^\"]+)\" \[ref={re.escape(target)}\](?: \[data-testid=\"([^\"]+)\"\])?"
            match = re.search(pattern, self.last_snapshot)
            if match:
                role, name, testid = match.groups()

                # Priority 1: Use data-testid if available (most unique)
                if testid:
                    return f"[data-testid='{testid}']"

                # Priority 2: Use role + name
                return f"internal:role={role}[name='{name}'i]"

        # If it's already a CSS selector or internal selector, return as is
        return target

    async def browser_navigate(self, url: str):
        page = await self._ensure_page()
        console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
        await page.goto(url)
        self.last_snapshot = None

    async def browser_navigate_back(self):
        page = await self._ensure_page()
        console.print("[bold blue]Going back[/bold blue]")
        await page.go_back()
        self.last_snapshot = None

    async def browser_type(
        self,
        target: str,
        text: str,
        element: Optional[str] = None,
        submit: Optional[bool] = False,
        slowly: Optional[bool] = False,
    ):
        """Type text into an element. No need to click first."""
        page = await self._ensure_page()
        resolved_target = self._resolve_target(target)
        console.print(f"[dim]Resolved target for type: {resolved_target}[/dim]")

        locator = page.locator(resolved_target)
        try:
            if slowly:
                await locator.press_sequentially(text)
            else:
                await locator.fill(text)

            if submit:
                await locator.press("Enter")
        except Exception as e:
            console.print(f"[bold red]Type failed on {resolved_target}: {e}[/bold red]")
            # Fallback: try finding by text content if role-based failed
            if resolved_target.startswith("internal:role"):
                name_match = re.search(r"name='([^']+)'", resolved_target)
                if name_match:
                    name = name_match.group(1)
                    console.print(
                        f"[dim]Fallback: Typing into first element with text '{name}'[/dim]"
                    )
                    await page.get_by_text(name, exact=True).first.fill(text)

    async def browser_press_key(self, key: str):
        page = await self._ensure_page()
        await page.keyboard.press(key)

    async def browser_hover(self, target: str, element: Optional[str] = None):
        page = await self._ensure_page()
        resolved_target = self._resolve_target(target)
        await page.locator(resolved_target).hover()

    async def browser_click(
        self,
        target: str,
        element: Optional[str] = None,
        doubleClick: Optional[bool] = False,
        button: Optional[Literal["left", "right", "middle"]] = "left",
        modifiers: Optional[List[str]] = None,
    ):
        page = await self._ensure_page()
        resolved_target = self._resolve_target(target)
        console.print(f"[dim]Resolved target for click: {resolved_target}[/dim]")

        locator = page.locator(resolved_target)
        options = {"button": button, "timeout": ACTION_TIMEOUT}
        if modifiers:
            options["modifiers"] = modifiers

        try:
            if doubleClick:
                await locator.dblclick(**options)
            else:
                await locator.click(**options)
        except Exception as e:
            console.print(
                f"[bold red]Click failed on {resolved_target}: {e}[/bold red]"
            )
            # Fallback strategies for strict mode violations or coverage issues
            if "strict mode violation" in str(e):
                console.print(
                    f"[dim]Fallback: Using .first() for {resolved_target}[/dim]"
                )
                await locator.first.click(**options)
            elif resolved_target.startswith("internal:role"):
                name_match = re.search(r"name='([^']+)'", resolved_target)
                if name_match:
                    name = name_match.group(1)
                    console.print(
                        f"[dim]Fallback: Clicking first element with text '{name}'[/dim]"
                    )
                    await page.get_by_text(name, exact=True).first.click()

    async def browser_fill_form(self, fields: List[dict]):
        page = await self._ensure_page()
        for field in fields:
            target = field.get("target")
            resolved_target = self._resolve_target(target)
            locator = page.locator(resolved_target)
            type_ = field.get("type")
            value = field.get("value")
            if type_ == "textbox":
                await locator.fill(value)
            elif type_ == "checkbox":
                if value.lower() == "true":
                    await locator.check()
                else:
                    await locator.uncheck()
            elif type_ == "combobox":
                await locator.select_option(value)

    async def browser_select_option(
        self, target: str, values: List[str], element: Optional[str] = None
    ):
        page = await self._ensure_page()
        resolved_target = self._resolve_target(target)
        await page.locator(resolved_target).select_option(values)

    async def browser_handle_dialog(
        self, accept: bool, promptText: Optional[str] = None
    ):
        page = await self._ensure_page()
        async with page.expect_dialog() as dialog_info:
            pass
        dialog = dialog_info.value
        if accept:
            await dialog.accept(promptText)
        else:
            await dialog.dismiss()

    async def browser_find(
        self,
        text: Optional[str] = None,
        regex: Optional[str] = None,
        max_results: Optional[int] = None,
        filename: Optional[str] = None,
    ):
        page = await self._ensure_page()
        content = await page.content()
        results = []
        if text and text.lower() in content.lower():
            results.append(f"Found '{text}'")
        if regex:
            import re as re_module

            matches = re_module.findall(regex, content)
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
        page = await self._ensure_page()
        if target:
            resolved_target = self._resolve_target(target)
            result = await page.locator(resolved_target).evaluate(function)
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
        page = await self._ensure_page()
        if filename:
            with open(self.output_dir / filename, "r") as f:
                code = f.read()
        return await page.evaluate(code)

    async def browser_snapshot(
        self,
        target: Optional[str] = None,
        filename: Optional[str] = None,
        depth: Optional[int] = None,
        boxes: Optional[bool] = False,
    ):
        page = await self._ensure_page()
        snapshot = await page.aria_snapshot(mode="ai", depth=depth, boxes=boxes)
        self.last_snapshot = snapshot

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
        page = await self._ensure_page()
        options = {"type": type, "scale": scale, "full_page": fullPage}
        if type == "jpeg":
            options["quality"] = 90

        if target:
            resolved_target = self._resolve_target(target)
            data = await page.locator(resolved_target).screenshot(**options)
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
        page = await self._ensure_page()
        if time:
            await page.wait_for_timeout(time * 1000)
        elif text:
            await page.get_by_text(text).first.wait_for(state="visible")
        elif textGone:
            await page.get_by_text(textGone).first.wait_for(state="hidden")

    async def browser_close(self):
        await self.close()


class PlaywrightToolRegistry:
    """Registry to map tool names to their schemas and implementations."""

    def __init__(self, toolbox: PlaywrightToolbox):
        self.toolbox = toolbox
        self.tools = {
            "browser_navigate": {
                "name": "browser_navigate",
                "description": "Navigate to a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to navigate to",
                        }
                    },
                    "required": ["url"],
                },
            },
            "browser_snapshot": {
                "name": "browser_snapshot",
                "description": "Capture accessibility snapshot of the current page. Use this to see the page structure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Optional filename to save snapshot",
                        }
                    },
                },
            },
            "browser_click": {
                "name": "browser_click",
                "description": "Perform click on a web page. Use the 'ref' value from the snapshot (e.g., 'e39') or a standard CSS selector.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Exact target element reference from the page snapshot (e.g., 'e39') or a CSS selector.",
                        },
                        "doubleClick": {
                            "type": "boolean",
                            "description": "Whether to double click",
                        },
                    },
                    "required": ["target"],
                },
            },
            "browser_type": {
                "name": "browser_type",
                "description": "Type text into an element. Use the 'ref' value from the snapshot (e.g., 'e39') or a standard CSS selector. No need to click first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Exact target element reference from the page snapshot (e.g., 'e39') or a CSS selector.",
                        },
                        "text": {"type": "string", "description": "Text to type"},
                        "submit": {
                            "type": "boolean",
                            "description": "Press Enter after typing",
                        },
                    },
                    "required": ["target", "text"],
                },
            },
            "browser_wait_for": {
                "name": "browser_wait_for",
                "description": "Wait for time or text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "number", "description": "Seconds to wait"},
                        "text": {"type": "string", "description": "Text to appear"},
                    },
                },
            },
        }

    def get_tool_schema(self) -> List[Dict]:
        return list(self.tools.values())

    async def execute_tool(self, name: str, arguments: Dict) -> Any:
        method = getattr(self.toolbox, name, None)
        if not method:
            raise ValueError(f"Tool {name} not found")
        return await method(**arguments)


class LocalAgent:
    """Manages the interaction between a local LLM and Playwright tools."""

    def __init__(self, api_base: str = LLM_BASE_URL, model: str = LLM_MODEL):
        self.client = openai.AsyncOpenAI(base_url=api_base, api_key="local")
        self.model = model
        self.registry: Optional[PlaywrightToolRegistry] = None
        self.history = []

    def set_registry(self, registry: PlaywrightToolRegistry):
        self.registry = registry

    async def think_and_act(
        self, goal: str, current_state: str, max_steps: int = 10
    ) -> str:
        if not self.registry:
            raise Exception("Tool registry not set")

        system_prompt = f"""
        You are an autonomous browser agent. Your goal is: {goal}
        
        Available tools:
        {json.dumps(self.registry.get_tool_schema(), indent=2)}

        IMPORTANT: 
        1. Always start by calling 'browser_snapshot' to see the current page structure.
        2. When interacting with elements (click, type), use the 'ref' value from the snapshot (e.g., 'e39') as the 'target'.
        3. Do NOT use CSS selectors like '.textbox[ref=e39]'. Just use 'e39'.
        4. For text inputs, use 'browser_type' directly. You do NOT need to click first.
        5. Respond with a JSON object containing:
           - "thought": Your reasoning about the next step.
           - "tool": The name of the tool to use.
           - "arguments": The arguments for the tool.
           - "finished": Boolean indicating if the goal is achieved.
        """

        for i in range(max_steps):
            console.print(f"[bold cyan]Step {i + 1}:[/bold cyan] Thinking...")
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Current state:\n{current_state}"},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    top_p=0.95,
                    presence_penalty=1.5,
                    stream_options={"include_usage": True},
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        },
                    },
                    stream=True,
                )
                content = ""
                async for part in stream:
                    if part.choices and part.choices[0].delta:
                        delta = part.choices[0].delta
                        if (
                            hasattr(delta, "reasoning_content")
                            and delta.reasoning_content
                        ):
                            print(delta.reasoning_content, end="", flush=True)
                        elif hasattr(delta, "content") and delta.content:
                            content += delta.content
                            print(delta.content, end="", flush=True)
                    usage = getattr(part, "usage", None)
                    if usage is not None:
                        print(
                            f"\n\nUsage Info:"
                            f"\nPrompt tokens: {usage.prompt_tokens}"
                            f"\nCompletion tokens: {usage.completion_tokens}"
                            f"\nTotal tokens: {usage.total_tokens}"
                        )

                console.print("\n")  # New line after stream
                decision = json.loads(content)
                console.print(f"[dim]Thought: {decision.get('thought')}[/dim]")

                if decision.get("finished"):
                    return "Goal achieved."

                tool_name = decision.get("tool")
                args = decision.get("arguments", {})

                if tool_name:
                    console.print(
                        f"[bold yellow]Executing:[/bold yellow] {tool_name}({args})"
                    )
                    result = await self.registry.execute_tool(tool_name, args)

                    # Always update state via snapshot after an action unless the action WAS a snapshot
                    if tool_name != "browser_snapshot":
                        current_state = await self.registry.toolbox.browser_snapshot()
                    else:
                        current_state = result

            except Exception as e:
                console.print(f"[bold red]Error in agent loop:[/bold red] {e}")
                break

        return "Max steps reached."


async def main():
    """Example usage of the Agent Loop"""
    toolbox = PlaywrightToolbox(headless=False)
    registry = PlaywrightToolRegistry(toolbox)
    agent = LocalAgent(model=LLM_MODEL)
    agent.set_registry(registry)

    try:
        await toolbox.start()
        await toolbox.browser_navigate(DEFAULT_URL)

        # Initial observation
        initial_snapshot = await toolbox.browser_snapshot(filename="initial.md")

        # Start the agent loop
        goal = "Find the search bar and type 'Software Developer'"
        await agent.think_and_act(goal, initial_snapshot)

    finally:
        await toolbox.close()


if __name__ == "__main__":
    asyncio.run(main())
