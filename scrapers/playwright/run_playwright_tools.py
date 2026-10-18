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
ACTION_TIMEOUT = 5000
NAVIGATION_TIMEOUT = 10000

SYSTEM_PROMPT_TEMPLATE = """
You are an autonomous browser agent powered by Playwright. Your goal is to: {goal}

Available Tools:
{tools_json}

CRITICAL RULES FOR INTERACTION:
1. USE REFS DIRECTLY: When the snapshot shows an element like `- textbox "Job title" [ref=e39]`, use "e39" as the target. Do NOT use CSS selectors like `[ref=e39]`.
2. NO UNNECESSARY CLICKS: For text inputs, use `browser_type` directly. It handles focusing and filling. Do NOT click before typing unless explicitly required for a non-input element.
3. HANDLE AMBIGUITY: If a role-based selector (e.g., `internal:role=link`) matches multiple elements, the tool will automatically pick the first one. You don't need to worry about strict mode violations.
4. OBSERVE FIRST: Always start with `browser_snapshot` if you haven't seen the page yet.
5. ONE STEP AT A TIME: Perform one action per step. Wait for the next snapshot to confirm the result.

RESPONSE FORMAT:
Return ONLY a valid JSON object with no markdown formatting:
{{
  "thought": "Brief reasoning about the current state and next action",
  "tool": "name_of_the_tool",
  "arguments": {{ ... }},
  "finished": false
}}
"""

USER_PROMPT_TEMPLATE = """
Current Page Snapshot:
{snapshot}

Previous Action Result:
{last_result}
"""


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
        console.print("[bold blue]Initializing Playwright...[/bold blue]")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        self.page = await self.context.new_page()
        self.page.set_default_timeout(ACTION_TIMEOUT)
        self.page.set_default_navigation_timeout(NAVIGATION_TIMEOUT)
        console.print("[green]Browser ready[/green]")

    async def close(self):
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
        return self.page

    def _resolve_target(self, target: str) -> str:
        if self.last_snapshot and re.match(r"^e\d+$", target):
            pattern = rf"- (\w+) \"([^\"]+)\" \[ref={re.escape(target)}\](?: \[data-testid=\"([^\"]+)\"\])?"
            match = re.search(pattern, self.last_snapshot)
            if match:
                role, name, testid = match.groups()
                if testid:
                    return f"[data-testid='{testid}']"
                return f"internal:role={role}[name='{name}'i]"
        return target

    async def browser_navigate(self, url: str):
        page = await self._ensure_page()
        console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
        await page.goto(url)
        self.last_snapshot = None

    async def browser_snapshot(
        self,
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

    async def browser_click(
        self,
        target: str,
        doubleClick: Optional[bool] = False,
        button: Literal["left", "right", "middle"] = "left",
    ):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        locator = page.locator(resolved).first  # Use .first to avoid strict mode errors
        try:
            if doubleClick:
                await locator.dblclick(button=button)
            else:
                await locator.click(button=button)
            console.print(f"[green]Clicked {target}[/green]")
        except Exception as e:
            console.print(f"[red]Click failed: {e}[/red]")

    async def browser_type(
        self, target: str, text: str, submit: Optional[bool] = False
    ):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        locator = page.locator(resolved).first
        try:
            await locator.fill(text)
            if submit:
                await locator.press("Enter")
            console.print(f"[green]Typed '{text}' into {target}[/green]")
        except Exception as e:
            console.print(f"[red]Type failed: {e}[/red]")

    async def browser_wait_for(
        self, time: Optional[float] = None, text: Optional[str] = None
    ):
        page = await self._ensure_page()
        if time:
            await page.wait_for_timeout(time * 1000)
        elif text:
            await page.get_by_text(text).first.wait_for(state="visible")

    async def browser_close(self):
        await self.close()


class PlaywrightToolRegistry:
    def __init__(self, toolbox: PlaywrightToolbox):
        self.toolbox = toolbox
        self.tools = {
            "browser_navigate": {
                "name": "browser_navigate",
                "description": "Navigate to URL",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
            "browser_snapshot": {
                "name": "browser_snapshot",
                "description": "Get page structure",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                },
            },
            "browser_click": {
                "name": "browser_click",
                "description": "Click element by ref",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "doubleClick": {"type": "boolean"},
                    },
                    "required": ["target"],
                },
            },
            "browser_type": {
                "name": "browser_type",
                "description": "Fill input by ref. No click needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "text": {"type": "string"},
                        "submit": {"type": "boolean"},
                    },
                    "required": ["target", "text"],
                },
            },
            "browser_wait_for": {
                "name": "browser_wait_for",
                "description": "Wait for time/text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "number"},
                        "text": {"type": "string"},
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
    def __init__(self, api_base: str = LLM_BASE_URL, model: str = LLM_MODEL):
        self.client = openai.AsyncOpenAI(base_url=api_base, api_key="local")
        self.model = model
        self.registry: Optional[PlaywrightToolRegistry] = None

    def set_registry(self, registry: PlaywrightToolRegistry):
        self.registry = registry

    async def think_and_act(
        self, goal: str, current_state: str, max_steps: int = 10
    ) -> str:
        if not self.registry:
            raise Exception("Registry not set")

        tools_json = json.dumps(self.registry.get_tool_schema(), indent=2)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(goal=goal, tools_json=tools_json)
        last_result = "None"

        for i in range(max_steps):
            console.print(f"[bold cyan]Step {i + 1}:[/bold cyan] Thinking...")
            user_prompt = USER_PROMPT_TEMPLATE.format(
                snapshot=current_state[:4000], last_result=last_result
            )

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )

                content = response.choices[0].message.content
                decision = json.loads(content)
                console.print(f"[dim]Thought: {decision.get('thought')}[/dim]")

                if decision.get("finished"):
                    return "Goal achieved."

                tool_name = decision.get("tool")
                args = decision.get("arguments", {})

                if tool_name:
                    console.print(f"[bold yellow]Executing:[/bold yellow] {tool_name}")
                    result = await self.registry.execute_tool(tool_name, args)
                    last_result = f"Executed {tool_name}"

                    # Always refresh state after action
                    current_state = await self.registry.toolbox.browser_snapshot()

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                last_result = f"Error: {str(e)}"

        return "Max steps reached."


async def main():
    toolbox = PlaywrightToolbox(headless=False)
    registry = PlaywrightToolRegistry(toolbox)
    agent = LocalAgent(model=LLM_MODEL)
    agent.set_registry(registry)

    try:
        await toolbox.start()
        await toolbox.browser_navigate(DEFAULT_URL)
        initial_snapshot = await toolbox.browser_snapshot(filename="initial.md")
        await agent.think_and_act(
            "Find the job search bar and type 'Software Developer'", initial_snapshot
        )
    finally:
        await toolbox.close()


if __name__ == "__main__":
    asyncio.run(main())
