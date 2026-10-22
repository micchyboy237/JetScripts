"""
Playwright JavaScript Evaluator CLI
Executes JavaScript expressions on live URLs or local HTML files using Playwright.
"""

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from jet.scrapers.browser.playwright import evaluate, parse_arg_value
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute JavaScript expressions on webpages or HTML files using Playwright."
    )
    parser.add_argument(
        "url_or_file",
        type=str,
        help="URL or path to an HTML file to evaluate against.",
    )
    parser.add_argument(
        "expression",
        type=str,
        help="JavaScript expression to evaluate.",
    )
    parser.add_argument(
        "-a",
        "--arg",
        action="append",
        dest="args",
        help="Arguments to pass to the JavaScript function. Can be specified multiple times.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to save the results (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Filename to save the result. Default: auto-generated based on input.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=30000,
        help="Maximum time in milliseconds to wait (default: 30000)",
    )
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (default: False)",
    )
    return parser.parse_args()


def format_result(result: any) -> str:
    """Format the result for saving to file."""
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2, default=str)
    return str(result)


def main():
    """Main entry point."""
    args = get_args()

    console.print(
        f"[bold blue]Cleaning output directory:[/bold blue] {args.output_dir}"
    )
    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parsed_args = None
    if args.args:
        parsed_args = [parse_arg_value(arg) for arg in args.args]

    try:
        console.print(f"[bold cyan]Processing input:[/bold cyan] {args.url_or_file}")

        result = asyncio.run(
            evaluate(
                source=args.url_or_file,
                expression=args.expression,
                args=parsed_args,
                timeout=args.timeout,
                headless=args.headless,
            )
        )

        input_path = Path(args.url_or_file)
        if args.filename:
            final_filename = args.filename
        else:
            if input_path.is_file():
                base = input_path.stem
            else:
                base = (
                    args.url_or_file.replace("://", "_")
                    .replace("/", "_")
                    .replace(":", "_")[:50]
                )
            final_filename = f"{base}_result.json"

        result_str = format_result(result)
        result_path = args.output_dir / final_filename

        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result_str)

        console.print(
            f"[green]💾 Result saved:[/green] "
            f"[link=file://{result_path.resolve()}]{result_path.name}[/link]"
        )

        console.print("[bold blue]Result:[/bold blue]")
        console.print(result_str)

    except Exception as e:
        console.print(f"[bold red]Failed to evaluate:[/bold red] {e}")
        raise


if __name__ == "__main__":
    main()
