import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import openai
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from pydantic import BaseModel, Field
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_URL = "https://ph.talent.com"
ACTION_TIMEOUT = 10000  # Increased timeout for stability
NAVIGATION_TIMEOUT = 15000

SYSTEM_PROMPT_TEMPLATE = """
You are an autonomous browser agent powered by Playwright. Your goal is to: {goal}

Available Tools:
{tools_json}

CRITICAL RULES FOR INTERACTION:
1. USE REFS DIRECTLY: When the snapshot shows an element like `- textbox "Job title" [ref=e39]`, use "e39" as the target.
2. HANDLE DIALOGS: If you see a cookie banner or dialog, use `browser_click` on its close button first.
3. NO UNNECESSARY CLICKS: For text inputs, use `browser_type` directly. It handles focusing and filling.
4. ONE STEP AT A TIME: Perform one action per step. 
5. REQUIRED TOOL: You MUST provide a `tool` name in your response unless `finished` is true. Do not just provide thoughts.

RESPONSE FORMAT:
You must respond using the provided JSON schema.
"""

USER_PROMPT_TEMPLATE = """
Current Page Snapshot:
{snapshot}

Previous Action Result:
{last_result}
"""


class NavigateArgs(BaseModel):
    url: str = Field(description="The URL to navigate to")


class SnapshotArgs(BaseModel):
    filename: Optional[str] = Field(
        None, description="Optional filename to save the snapshot"
    )
    depth: Optional[int] = Field(None, description="Depth of the snapshot tree")
    boxes: Optional[bool] = Field(False, description="Include bounding boxes")


class ClickArgs(BaseModel):
    target: str = Field(description="Target element ref (e.g., e39) or selector")
    doubleClick: Optional[bool] = Field(False, description="Whether to double click")
    button: Literal["left", "right", "middle"] = Field(
        "left", description="Mouse button to use"
    )


class TypeArgs(BaseModel):
    target: str = Field(description="Target element ref (e.g., e39) or selector")
    text: str = Field(description="Text to type")
    submit: Optional[bool] = Field(
        False, description="Whether to press Enter after typing"
    )


class WaitForArgs(BaseModel):
    time: Optional[float] = Field(None, description="Time to wait in seconds")
    text: Optional[str] = Field(None, description="Text to wait for")


class PressKeyArgs(BaseModel):
    key: str = Field(description="Name of the key to press")


class HoverArgs(BaseModel):
    target: str = Field(description="Target element ref (e.g., e39) or selector")


class FindArgs(BaseModel):
    text: Optional[str] = Field(None, description="Plain text to search for")
    regex: Optional[str] = Field(None, description="Regular expression to search for")
    maxResults: Optional[int] = Field(None, description="Max results")


class FillFormArgs(BaseModel):
    fields: List[Dict[str, Any]] = Field(description="Fields to fill in")


class SelectOptionArgs(BaseModel):
    target: str = Field(description="Target element ref")
    values: List[str] = Field(description="Values to select")


class HandleDialogArgs(BaseModel):
    accept: bool = Field(description="Whether to accept the dialog")
    promptText: Optional[str] = Field(None, description="Prompt text")


class EvaluateArgs(BaseModel):
    function: str = Field(description="JS function to evaluate")
    target: Optional[str] = Field(None, description="Target element ref")


class RunCodeArgs(BaseModel):
    code: str = Field(description="JS code to execute")


class ScreenshotArgs(BaseModel):
    filename: Optional[str] = Field(None, description="Filename")
    fullPage: Optional[bool] = Field(False, description="Full page screenshot")
    type: Literal["png", "jpeg", "webp"] = Field("png", description="Image format")


class AgentDecision(BaseModel):
    thought: str = Field(description="Brief reasoning")
    tool: str = Field(description="Name of the tool to execute")  # Made required
    arguments: Optional[Dict[str, Any]] = Field(
        None, description="Arguments for the tool"
    )
    finished: bool = Field(False, description="Whether the goal has been achieved")


def _get_decision_schema() -> Dict[str, Any]:
    schema = AgentDecision.model_json_schema()
    return {
        "name": "AgentDecision",
        "strict": True,
        "schema": schema,
    }


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
        # Try to resolve ref from last snapshot
        if self.last_snapshot and re.match(r"^e\d+$", target):
            pattern = rf"- (\w+) \"([^\"]+)\" \[ref={re.escape(target)}\]"
            match = re.search(pattern, self.last_snapshot)
            if match:
                role, name = match.groups()
                return f"internal:role={role}[name='{name}'i]"

        # Fallback: If it looks like a ref but wasn't found, try aria-ref selector
        if re.match(r"^e\d+$", target):
            return f"aria-ref={target}"

        return target

    async def browser_navigate(self, url: str):
        page = await self._ensure_page()
        console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
        await page.goto(url)
        self.last_snapshot = None

    async def browser_navigate_back(self):
        page = await self._ensure_page()
        await page.go_back()
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
        console.print(f"[green]Snapshot saved to {output_path.name}[/green]")
        return snapshot

    async def browser_click(
        self,
        target: str,
        doubleClick: Optional[bool] = False,
        button: Literal["left", "right", "middle"] = "left",
    ):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        locator = page.locator(resolved).first
        try:
            await locator.wait_for(state="visible", timeout=5000)
            if doubleClick:
                await locator.dblclick(button=button)
            else:
                await locator.click(button=button)
            console.print(f"[green]Clicked {target}[/green]")
        except Exception as e:
            console.print(f"[red]Click failed: {e}[/red]")
            raise e

    async def browser_type(
        self, target: str, text: str, submit: Optional[bool] = False
    ):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        locator = page.locator(resolved).first
        try:
            await locator.wait_for(state="visible", timeout=5000)
            await locator.fill(text)
            if submit:
                await locator.press("Enter")
            console.print(f"[green]Typed '{text}' into {target}[/green]")
        except Exception as e:
            console.print(f"[red]Type failed: {e}[/red]")
            raise e

    async def browser_wait_for(
        self, time: Optional[float] = None, text: Optional[str] = None
    ):
        page = await self._ensure_page()
        if time:
            await page.wait_for_timeout(time * 1000)
        elif text:
            await page.get_by_text(text).first.wait_for(state="visible")

    async def browser_press_key(self, key: str):
        page = await self._ensure_page()
        await page.keyboard.press(key)
        console.print(f"[green]Pressed key: {key}[/green]")

    async def browser_hover(self, target: str):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        locator = page.locator(resolved).first
        await locator.hover()
        console.print(f"[green]Hovered {target}[/green]")

    async def browser_find(
        self,
        text: Optional[str] = None,
        regex: Optional[str] = None,
        maxResults: Optional[int] = None,
    ):
        if self.last_snapshot:
            console.print(f"[dim]Searching snapshot for: {text or regex}[/dim]")
            return "Search completed in snapshot."
        return "No snapshot available."

    async def browser_fill_form(self, fields: List[Dict[str, Any]]):
        page = await self._ensure_page()
        for field in fields:
            target = field.get("target")
            value = field.get("value")
            if target and value is not None:
                resolved = self._resolve_target(target)
                locator = page.locator(resolved).first
                await locator.fill(str(value))
                console.print(f"[green]Filled {target} with {value}[/green]")

    async def browser_select_option(self, target: str, values: List[str]):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        locator = page.locator(resolved).first
        await locator.select_option(values)
        console.print(f"[green]Selected options {values} in {target}[/green]")

    async def browser_handle_dialog(
        self, accept: bool, promptText: Optional[str] = None
    ):
        page = await self._ensure_page()

        async def handle_dialog(dialog):
            if accept:
                await dialog.accept(promptText)
            else:
                await dialog.dismiss()

        page.once("dialog", handle_dialog)
        console.print(f"[green]Set dialog handler: accept={accept}[/green]")

    async def browser_evaluate(self, function: str, target: Optional[str] = None):
        page = await self._ensure_page()
        if target:
            resolved = self._resolve_target(target)
            locator = page.locator(resolved).first
            result = await locator.evaluate(function)
        else:
            result = await page.evaluate(function)
        console.print(f"[green]Evaluated JS. Result: {result}[/green]")
        return result

    async def browser_run_code_unsafe(self, code: str):
        page = await self._ensure_page()
        script = f"(async () => {{ {code} }})()"
        try:
            result = await page.evaluate(script)
            console.print(f"[green]Executed custom code. Result: {result}[/green]")
            return result
        except Exception as e:
            console.print(f"[red]Code execution failed: {e}[/red]")
            raise e

    async def browser_take_screenshot(
        self, filename: Optional[str] = None, fullPage: bool = False, type: str = "png"
    ):
        page = await self._ensure_page()
        output_path = self.output_dir / (filename or f"screenshot.{type}")
        await page.screenshot(path=str(output_path), full_page=fullPage, type=type)
        console.print(f"[green]Screenshot saved to {output_path.name}[/green]")
        return str(output_path)

    async def browser_close(self):
        await self.close()


class PlaywrightToolRegistry:
    def __init__(self, toolbox: PlaywrightToolbox):
        self.toolbox = toolbox
        self.tool_map = {
            "browser_navigate": {
                "model": NavigateArgs,
                "desc": "Navigate to a specific URL",
            },
            "browser_navigate_back": {"model": BaseModel, "desc": "Go back"},
            "browser_snapshot": {"model": SnapshotArgs, "desc": "Get page snapshot"},
            "browser_click": {"model": ClickArgs, "desc": "Click an element"},
            "browser_type": {"model": TypeArgs, "desc": "Type text into an input"},
            "browser_wait_for": {"model": WaitForArgs, "desc": "Wait for time/text"},
            "browser_press_key": {"model": PressKeyArgs, "desc": "Press a key"},
            "browser_hover": {"model": HoverArgs, "desc": "Hover over an element"},
            "browser_find": {"model": FindArgs, "desc": "Find text in snapshot"},
            "browser_fill_form": {
                "model": FillFormArgs,
                "desc": "Fill multiple fields",
            },
            "browser_select_option": {
                "model": SelectOptionArgs,
                "desc": "Select dropdown option",
            },
            "browser_handle_dialog": {
                "model": HandleDialogArgs,
                "desc": "Handle dialog",
            },
            "browser_evaluate": {"model": EvaluateArgs, "desc": "Evaluate JS"},
            "browser_run_code_unsafe": {
                "model": RunCodeArgs,
                "desc": "Run unsafe code",
            },
            "browser_take_screenshot": {
                "model": ScreenshotArgs,
                "desc": "Take screenshot",
            },
            "browser_close": {"model": BaseModel, "desc": "Close browser"},
        }

    def get_tool_schema(self) -> List[Dict]:
        schemas = []
        for name, info in self.tool_map.items():
            model = info["model"]
            schema = (
                {"type": "object", "properties": {}}
                if model == BaseModel
                else model.model_json_schema()
            )
            schemas.append(
                {"name": name, "description": info["desc"], "parameters": schema}
            )
        return schemas

    async def execute_tool(self, name: str, arguments: Dict) -> Any:
        if name not in self.tool_map:
            raise ValueError(f"Tool {name} not found")

        model_class = self.tool_map[name]["model"]
        validated_args = (
            {}
            if model_class == BaseModel
            else model_class(**arguments).model_dump(exclude_none=True)
        )

        method = getattr(self.toolbox, name, None)
        if not method:
            raise ValueError(f"Method {name} not implemented")

        return await method(**validated_args)


class LocalAgent:
    def __init__(self, api_base: str = LLM_BASE_URL, model: str = LLM_MODEL):
        self.client = openai.AsyncOpenAI(base_url=api_base, api_key="local")
        self.model = model
        self.registry: Optional[PlaywrightToolRegistry] = None

    def set_registry(self, registry: PlaywrightToolRegistry):
        self.registry = registry

    async def think_and_act(
        self, goal: str, current_state: str, max_steps: int = 10, **kwargs
    ) -> str:
        if not self.registry:
            raise Exception("Registry not set")

        tools_json = json.dumps(self.registry.get_tool_schema(), indent=2)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(goal=goal, tools_json=tools_json)
        last_result = "None"

        decision_schema = _get_decision_schema()
        response_format = {"type": "json_schema", "json_schema": decision_schema}

        for i in range(max_steps):
            console.print(f"[bold cyan]Step {i + 1}:[/bold cyan] Thinking...")
            user_prompt = USER_PROMPT_TEMPLATE.format(
                snapshot=current_state[:4000], last_result=last_result
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    stream=True,
                    response_format=response_format,
                    **kwargs,
                )
                content = ""
                async for part in stream:
                    if part.choices and part.choices[0].delta.content:
                        content += part.choices[0].delta.content
                        print(part.choices[0].delta.content, end="", flush=True)

                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                decision = AgentDecision.model_validate_json(content)
                console.print(f"\n[dim]Thought: {decision.thought}[/dim]")

                if decision.finished:
                    return "Goal achieved."

                console.print(f"[bold yellow]Executing:[/bold yellow] {decision.tool}")
                try:
                    await self.registry.execute_tool(
                        decision.tool, decision.arguments or {}
                    )
                    last_result = f"Successfully executed {decision.tool}"
                    # Always update snapshot after successful action
                    current_state = await self.registry.toolbox.browser_snapshot()
                except Exception as exec_err:
                    console.print(f"[red]Execution error: {exec_err}[/red]")
                    last_result = f"Error executing {decision.tool}: {str(exec_err)}"
                    # On error, take a snapshot anyway to see if the page changed (e.g. error modal)
                    current_state = await self.registry.toolbox.browser_snapshot()

            except Exception as e:
                console.print(f"[bold red]API Error:[/bold red] {e}")
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
            "Find the job search bar, type 'Software Developer', select 'Muntinlupa City' in location, and click Find jobs.",
            initial_snapshot,
        )
    finally:
        await toolbox.close()


if __name__ == "__main__":
    asyncio.run(main())
