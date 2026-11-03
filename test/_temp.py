import asyncio
import shutil
from pathlib import Path
from typing import List, Optional

from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.scrapers.browser.playwright_helpers import ascrape_urls
from rich.console import Console
from rich.tree import Tree

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_job_posting_from_html(html_content: str) -> str:
    """
    Extracts job posting details using trafilatura's extract function.

    Args:
        html_content: The HTML string of the job posting page.
    Returns:
        The extracted markdown content.
    """

    md_content = convert_html_to_markdown(html_content, ignore_links=False)
    return md_content


def save_job_data(
    url: str,
    markdown_content: str,
    html_content: str,
    index: int,
    screenshot: Optional[bytes] = None,
) -> Path:
    """
    Saves the extracted job data (MD, HTML, and screenshot) to a file in the OUTPUT_DIR.
    """
    sub_dir = OUTPUT_DIR / f"url_{index}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    safe_name = (
        url.replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        .replace("?", "_")
        .replace("&", "_")
        .replace("=", "_")
    )
    if len(safe_name) > 150:
        safe_name = safe_name[:150]
    md_path = sub_dir / f"{safe_name}.md"
    html_path = sub_dir / f"{safe_name}.html"
    md_path.write_text(markdown_content, encoding="utf-8")
    html_path.write_text(html_content, encoding="utf-8")
    console.print(f"[green]✓ Saved:[/green] {md_path.name}")
    console.print(f"[dim]  ↳ HTML: {html_path.name}[/dim]")
    if screenshot:
        screenshot_path = sub_dir / f"{safe_name}.png"
        screenshot_path.write_bytes(screenshot)
        console.print(f"[dim]  ↳ Screenshot: {screenshot_path.name}[/dim]")
    return sub_dir


async def main():
    urls = [
        "https://www.linkedin.com/jobs/view/4484950054",
        "https://www.onlinejobs.ph/jobseekers/job/1759438",
        "https://ph.jobstreet.com/job/95517752",
        "https://ph.indeed.com/viewjob?jk=c1f6a3f04bf2d239",
    ]
    console.print(f"[bold cyan]Starting extraction for {len(urls)} jobs...[/bold cyan]")
    console.print(f"[dim]Output Directory: {OUTPUT_DIR}[/dim]\n")
    saved_dirs: List[Path] = []
    async for result in ascrape_urls(
        urls=urls,
        num_parallel=2,
        headless=False,
        show_progress=True,
        with_screenshot=True,
    ):
        if result["status"] == "completed" and result["html"]:
            url = result["url"]
            html = result["html"]
            screenshot = result.get("screenshot")
            try:
                index = urls.index(url)
            except ValueError:
                index = len(saved_dirs)
            console.print(f"\n[yellow]Processing {url}...[/yellow]")
            markdown_content = extract_job_posting_from_html(html)
            if markdown_content:
                saved_path = save_job_data(
                    url, markdown_content, html, index, screenshot
                )
                saved_dirs.append(saved_path)
                preview = markdown_content[:200].replace("\n", " ")
                console.print(f"[dim]Preview: {preview}...[/dim]")
            else:
                console.print(f"[red]✗ Extraction failed for {url}[/red]")
                sub_dir = OUTPUT_DIR / f"url_{index}_failed"
                sub_dir.mkdir(parents=True, exist_ok=True)
                safe_name = (
                    url.replace("https://", "")
                    .replace("/", "_")
                    .replace("?", "_")[:100]
                )
                (sub_dir / f"{safe_name}.html").write_text(html, encoding="utf-8")
                if screenshot:
                    (sub_dir / f"{safe_name}.png").write_bytes(screenshot)
        elif result["status"] != "started":
            console.print(
                f"[red]✗ Failed to scrape {result['url']}: {result['status']}[/red]"
            )
    console.print("\n[bold green]Extraction Complete. Resource Tree:[/bold green]")
    root = Tree(OUTPUT_DIR.name)
    for subdir in sorted(OUTPUT_DIR.iterdir()):
        if subdir.is_dir():
            branch = root.add(f"[bold blue]{subdir.name}[/bold blue]")
            for file in sorted(subdir.iterdir()):
                icon = "📄"
                if file.suffix == ".html":
                    icon = "🌐"
                elif file.suffix == ".md":
                    icon = "📝"
                elif file.suffix == ".png":
                    icon = "🖼️"
                file_link = f"file://{file.resolve()}"
                branch.add(f"{icon} [link={file_link}]{file.name}[/link]")
    console.print(root)


if __name__ == "__main__":
    asyncio.run(main())
