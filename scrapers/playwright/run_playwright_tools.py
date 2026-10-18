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
You must respond using the provided JSON schema. Do not include markdown formatting or extra text outside the JSON object.
"""

USER_PROMPT_TEMPLATE = """
Current Page Snapshot:
{snapshot}

Previous Action Result:
{last_result}
"""

# --- Argument Models ---


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
    key: str = Field(
        description="Name of the key to press or a character to generate, such as ArrowLeft or a"
    )


class HoverArgs(BaseModel):
    target: str = Field(description="Target element ref (e.g., e39) or selector")


class FindArgs(BaseModel):
    text: Optional[str] = Field(
        None, description="Plain text to search for in the page snapshot"
    )
    regex: Optional[str] = Field(
        None, description="Regular expression to search for in the page snapshot"
    )
    maxResults: Optional[int] = Field(
        None, description="Maximum number of matching items to return"
    )


class FillFormArgs(BaseModel):
    fields: List[Dict[str, Any]] = Field(
        description="Fields to fill in. Each field should have 'target', 'name', 'type' (textbox/checkbox/radio/combobox/slider), and 'value'."
    )


class SelectOptionArgs(BaseModel):
    target: str = Field(description="Target element ref (e.g., e39) or selector")
    values: List[str] = Field(description="Array of values to select in the dropdown.")


class HandleDialogArgs(BaseModel):
    accept: bool = Field(description="Whether to accept the dialog.")
    promptText: Optional[str] = Field(
        None, description="The text of the prompt in case of a prompt dialog."
    )


class EvaluateArgs(BaseModel):
    function: str = Field(
        description="JavaScript function to evaluate. e.g., () => document.title"
    )
    target: Optional[str] = Field(
        None, description="Target element ref if evaluation is on an element"
    )


class RunCodeArgs(BaseModel):
    code: str = Field(
        description="A JavaScript function containing Playwright code to execute. e.g., async (page) => { return await page.title(); }"
    )


class ScreenshotArgs(BaseModel):
    filename: Optional[str] = Field(
        None, description="File name to save the screenshot"
    )
    fullPage: Optional[bool] = Field(
        False, description="Take screenshot of full scrollable page"
    )
    type: Literal["png", "jpeg", "webp"] = Field("png", description="Image format")


# --- Decision Model ---


class AgentDecision(BaseModel):
    thought: str = Field(
        description="Brief reasoning about the current state and next action"
    )
    tool: Optional[str] = Field(None, description="Name of the tool to execute")
    arguments: Optional[Dict[str, Any]] = Field(
        None, description="Arguments for the tool"
    )
    finished: bool = Field(False, description="Whether the goal has been achieved")


def _get_decision_schema() -> Dict[str, Any]:
    """Generate a strict JSON schema from the AgentDecision Pydantic model."""
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
        locator = page.locator(resolved).first
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
        # Note: This is a simplified implementation. Real implementation would parse the snapshot.
        if self.last_snapshot:
            console.print(f"[dim]Searching snapshot for: {text or regex}[/dim]")
            # In a real scenario, this might return specific refs or coordinates
            return "Search completed in snapshot."
        return "No snapshot available to search."

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

        # Note: Dialogs are usually handled via event listeners in Playwright.
        # This tool assumes a dialog is already pending or sets up a one-time handler.
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
        # Wrapping in an IIFE to allow arbitrary code execution
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
        await page.screenshot(path=str(output_path), full_page=fullPage, type=type)  # type: ignore
        console.print(f"[green]Screenshot saved to {output_path.name}[/green]")
        return str(output_path)

    async def browser_close(self):
        await self.close()


class PlaywrightToolRegistry:
    def __init__(self, toolbox: PlaywrightToolbox):
        self.toolbox = toolbox
        # Map tool names to their argument models and descriptions
        self.tool_map = {
            "browser_navigate": {
                "model": NavigateArgs,
                "desc": "Navigate to a specific URL",
            },
            "browser_navigate_back": {
                "model": BaseModel,
                "desc": "Go back to the previous page",
            },
            "browser_snapshot": {
                "model": SnapshotArgs,
                "desc": "Get the accessibility snapshot of the page",
            },
            "browser_click": {
                "model": ClickArgs,
                "desc": "Click an element on the page",
            },
            "browser_type": {
                "model": TypeArgs,
                "desc": "Type text into an input field",
            },
            "browser_wait_for": {
                "model": WaitForArgs,
                "desc": "Wait for a specific time or text to appear",
            },
            "browser_press_key": {
                "model": PressKeyArgs,
                "desc": "Press a key on the keyboard",
            },
            "browser_hover": {"model": HoverArgs, "desc": "Hover over an element"},
            "browser_find": {
                "model": FindArgs,
                "desc": "Find text in the current snapshot",
            },
            "browser_fill_form": {
                "model": FillFormArgs,
                "desc": "Fill multiple form fields at once",
            },
            "browser_select_option": {
                "model": SelectOptionArgs,
                "desc": "Select options in a dropdown",
            },
            "browser_handle_dialog": {
                "model": HandleDialogArgs,
                "desc": "Accept or dismiss a dialog",
            },
            "browser_evaluate": {
                "model": EvaluateArgs,
                "desc": "Evaluate JavaScript on the page or element",
            },
            "browser_run_code_unsafe": {
                "model": RunCodeArgs,
                "desc": "Run arbitrary Playwright code (unsafe)",
            },
            "browser_take_screenshot": {
                "model": ScreenshotArgs,
                "desc": "Take a screenshot of the page",
            },
            "browser_close": {"model": BaseModel, "desc": "Close the browser"},
        }

    def get_tool_schema(self) -> List[Dict]:
        schemas = []
        for name, info in self.tool_map.items():
            model = info["model"]
            # Handle BaseModel (no args) vs specific args
            if model == BaseModel:
                schema = {"type": "object", "properties": {}}
            else:
                schema = model.model_json_schema()

            schemas.append(
                {"name": name, "description": info["desc"], "parameters": schema}
            )
        return schemas

    async def execute_tool(self, name: str, arguments: Dict) -> Any:
        if name not in self.tool_map:
            raise ValueError(f"Tool {name} not found")

        model_class = self.tool_map[name]["model"]
        try:
            # Validate arguments using Pydantic
            if model_class == BaseModel:
                validated_args = {}
            else:
                validated_args = model_class(**arguments).model_dump(exclude_none=True)
        except Exception as e:
            raise ValueError(f"Invalid arguments for {name}: {str(e)}")

        method = getattr(self.toolbox, name, None)
        if not method:
            raise ValueError(f"Method {name} not implemented in toolbox")

        # Pass validated model fields as kwargs
        return await method(**validated_args)


class LocalAgent:
    def __init__(self, api_base: str = LLM_BASE_URL, model: str = LLM_MODEL):
        self.client = openai.AsyncOpenAI(base_url=api_base, api_key="local")
        self.model = model
        self.registry: Optional[PlaywrightToolRegistry] = None

    def set_registry(self, registry: PlaywrightToolRegistry):
        self.registry = registry

    async def think_and_act(
        self,
        goal: str,
        current_state: str,
        max_steps: int = 10,
        **kwargs,
    ) -> str:
        if not self.registry:
            raise Exception("Registry not set")

        tools_json = json.dumps(self.registry.get_tool_schema(), indent=2)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(goal=goal, tools_json=tools_json)
        last_result = "None"

        # Generate strict schema for response format
        decision_schema = _get_decision_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": decision_schema,
        }

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
                create_kwargs = dict(
                    model=self.model,
                    messages=messages,
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
                    response_format=response_format,
                    **kwargs,
                )
                stream = await self.client.chat.completions.create(**create_kwargs)
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

                # Clean content just in case
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                # Use Pydantic to parse and validate the decision
                try:
                    decision = AgentDecision.model_validate_json(content)
                except Exception as parse_err:
                    console.print(
                        f"[red]Failed to parse LLM response: {parse_err}[/red]"
                    )
                    console.print(f"[dim]Raw content: {content[:200]}...[/dim]")
                    last_result = f"Error: Invalid response format from LLM"
                    continue

                console.print(f"[dim]Thought: {decision.thought}[/dim]")

                if decision.finished:
                    return "Goal achieved."

                if decision.tool:
                    console.print(
                        f"[bold yellow]Executing:[/bold yellow] {decision.tool}"
                    )
                    try:
                        result = await self.registry.execute_tool(
                            decision.tool, decision.arguments or {}
                        )
                        last_result = f"Successfully executed {decision.tool}"
                    except Exception as exec_err:
                        console.print(f"[red]Execution error: {exec_err}[/red]")
                        last_result = (
                            f"Error executing {decision.tool}: {str(exec_err)}"
                        )

                    # Always take a new snapshot after an action to update state
                    if decision.tool != "browser_close":
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
            """
Find the job search input, type 'Software Developer'.
Select 'Muntinlupa City'.
Submit search.
            """.strip(),
            initial_snapshot,
        )
    finally:
        await toolbox.close()


if __name__ == "__main__":
    asyncio.run(main())
