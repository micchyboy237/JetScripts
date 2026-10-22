"""
Autonomous Playwright Browser Agent - Entry Point
Orchestrates the agent workflow based on CLI arguments.
Usage Examples:
python run_playwright_agent.py https://ph.indeed.com -g "
1. Search 'AI Engineer' in 'Remote' location, and click 'Find jobs' button.
2. Display all the job titles and links in the first page as JSON.
"
"""

import asyncio
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from jet.scrapers.browser.agent.core import LocalAgent
from jet.scrapers.browser.agent.registry import PlaywrightToolRegistry
from jet.scrapers.browser.agent.toolbox import PlaywrightToolbox
from jet.scrapers.browser.agent.utils import get_step_dir, save_json
from jet_telemetry import chain, get_trace_url, initialize_telemetry
from rich.console import Console

console = Console()
DEFAULT_INPUT = "https://ph.talent.com"
DEFAULT_GOAL = """
1. Search 'AI Engineer' in 'Remote' location, and click 'Find jobs' button.
2. Display all the result links in the first page
"""
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Playwright Browser Agent")
    parser.add_argument(
        "url_or_file",
        type=str,
        nargs="?",
        default=DEFAULT_INPUT,
        help="URL or path to an HTML file",
    )
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "-g", "--goal", type=str, default=DEFAULT_GOAL, help="The goal for the agent"
    )
    parser.add_argument(
        "-n", "--max-steps", type=int, default=15, help="Maximum number of steps"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Directory for output files",
    )
    parser.add_argument(
        "-a",
        "--api-base",
        type=str,
        default=LLM_BASE_URL,
        help="Base URL for the LLM API",
    )
    parser.add_argument(
        "-m", "--model", type=str, default=LLM_MODEL, help="Model name to use"
    )
    parser.add_argument(
        "-mmt",
        "--max-message-tokens",
        type=int,
        default=12000,
        help="Maximum tokens of messages to send to LLM",
    )
    parser.add_argument(
        "-mst",
        "--max-snapshot-tokens",
        type=int,
        default=3000,
        help="Maximum tokens of snapshot to send to LLM",
    )
    parser.add_argument(
        "-s",
        "--auto-scroll",
        action="store_true",
        default=True,
        help="Automatically scroll to bottom after page load to trigger lazy loading",
    )
    parser.add_argument(
        "-S",
        "--max-scrolls",
        type=int,
        default=5,
        help="Maximum number of scroll jumps when auto-scrolling",
    )
    parser.add_argument(
        "-d",
        "--scroll-delay",
        type=int,
        default=1400,
        help="Delay between scrolls in ms",
    )
    parser.add_argument(
        "-t",
        "--scroll-timeout",
        type=int,
        default=45000,
        help="Overall scroll timeout in ms",
    )
    parser.add_argument(
        "-M",
        "--scroll-mode",
        type=str,
        choices=["jump", "increment"],
        default="increment",
        help="Scroll mode: 'jump' to bottom or 'increment' by viewport",
    )
    return parser.parse_args()


@chain(name="playwright-agent-entry")
async def main():
    args = get_args()
    initialize_telemetry(service_name="playwright-agent")
    console.print("[bold cyan]🚀 Starting Playwright Agent...[/bold cyan]")
    trace_url = get_trace_url()
    if trace_url:
        console.print(f"[dim]Trace URL: {trace_url}[/dim]")
    output_dir = args.output_dir
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    toolbox = PlaywrightToolbox(
        headless=args.headless,
        output_dir=output_dir,
        auto_scroll=args.auto_scroll,
        max_scrolls=args.max_scrolls,
        scroll_delay_ms=args.scroll_delay,
        scroll_timeout_ms=args.scroll_timeout,
        scroll_mode=args.scroll_mode,
    )
    registry = PlaywrightToolRegistry(toolbox)
    agent = LocalAgent(
        api_base=args.api_base,
        model=args.model,
        registry=registry,
        output_dir=output_dir,
        max_message_tokens=args.max_message_tokens,
        max_snapshot_tokens=args.max_snapshot_tokens,
    )
    try:
        await toolbox.start()
        step_0_dir = get_step_dir(0, output_dir)
        input_source = args.url_or_file
        input_path = Path(input_source)
        if input_path.is_file():
            console.print(
                f"[bold blue]Reading HTML from file:[/bold blue] {input_path}"
            )
            with open(input_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            await toolbox.browser_set_content(html_content)
            save_json(
                step_0_dir / "action.json",
                {"action": "set_content", "file": str(input_path)},
            )
        else:
            console.print(f"[bold blue]Navigating to URL:[/bold blue] {input_source}")
            await toolbox.browser_navigate(input_source)
            save_json(
                step_0_dir / "action.json", {"action": "navigate", "url": input_source}
            )
        await toolbox.browser_take_screenshot(
            filename="initial_screenshot.png", step_dir=step_0_dir
        )
        initial_snapshot = await toolbox.browser_snapshot(
            filename="initial_snapshot.md", step_dir=step_0_dir
        )
        await agent.think_and_act(args.goal, initial_snapshot, max_steps=args.max_steps)
    finally:
        if not toolbox.is_closed:
            await toolbox.close()
    console.print("[bold green]✅ Agent Finished.[/bold green]")
    final_trace_url = get_trace_url()
    if final_trace_url:
        console.print(
            f"[bold][link={final_trace_url}]View Full Trace in Phoenix[/link][/bold]"
        )


if __name__ == "__main__":
    asyncio.run(main())
