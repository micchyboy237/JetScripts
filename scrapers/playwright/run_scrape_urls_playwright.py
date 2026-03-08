import os
import shutil
import time
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

from jet.code.markdown_utils import (
    analyze_markdown,
    base_parse_markdown,
    convert_html_to_markdown,
    derive_by_header_hierarchy,
)
from jet.code.markdown_utils._preprocessors import extract_markdown_links
from jet.file.utils import save_file
from jet.logger import logger
from jet.scrapers.playwright_utils import scrape_urls, scrape_urls_sync
from jet.scrapers.utils import scrape_links
from jet.utils.text import format_sub_dir, format_sub_source_dir
from unstructured.partition.html import partition_html

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.orange(f"Log file: {log_file}")


class ScrapeResult(TypedDict):
    url: str
    status: Literal["started", "completed", "failed_no_html", "failed_error"]
    html: Optional[str]
    screenshot: Optional[bytes]


async def async_example(urls: List[str]) -> None:
    sub_dir = f"{OUTPUT_DIR}/async_results"

    screenshots_dir = os.path.join(sub_dir, "screenshots")
    html_files_dir = os.path.join(sub_dir, "html_files")
    os.makedirs(screenshots_dir, exist_ok=True)
    os.makedirs(html_files_dir, exist_ok=True)

    html_list = []

    start = time.perf_counter()
    async for result in scrape_urls(
        urls,
        num_parallel=3,
        limit=5,
        show_progress=True,
        timeout=5000,
        max_retries=3,
        with_screenshot=True,
        headless=False,
        wait_for_js=False,
        use_cache=False,
        scroll_strategy="until_stable",
        scroll_mode="increment",
    ):
        if result["status"] == "completed" and result["html"]:
            all_links = scrape_links(result["html"], base_url=result["url"])
            safe_filename = format_sub_dir(result["url"])

            if result["screenshot"]:
                screenshot_path = os.path.join(screenshots_dir, f"{safe_filename}.png")
                with open(screenshot_path, "wb") as f:
                    f.write(result["screenshot"])
                logger.success(
                    f"Scraped {result['url']}, links count: {len(all_links)}, "
                    f"screenshot saved: {screenshot_path}"
                )
            else:
                logger.success(
                    f"Scraped {result['url']}, links count: {len(all_links)}, screenshot: not taken"
                )

            html_path = os.path.join(html_files_dir, f"{safe_filename}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(result["html"])
            logger.success(f"Saved HTML for {result['url']} to: {html_path}")
            html_list.append(result["html"])
    duration = time.perf_counter() - start
    logger.info(f"Done async scraped {len(html_list)} htmls in {duration:.2f} seconds")


def sync_example(urls: List[str]) -> None:
    html_list = []

    start = time.perf_counter()
    results = scrape_urls_sync(
        urls,
        num_parallel=3,
        limit=5,
        show_progress=True,
        timeout=5000,
        max_retries=3,
        with_screenshot=True,
        headless=False,
        wait_for_js=False,
        use_cache=False,
        scroll_strategy="until_stable",
        scroll_mode="increment",
    )

    for result in results:
        html = result["html"]
        if result["status"] == "completed" and html:
            url = result["url"]
            sub_output_dir = OUTPUT_DIR / format_sub_source_dir(url) / "sync_results"
            shutil.rmtree(sub_output_dir, ignore_errors=True)
            sub_output_dir.mkdir(parents=True, exist_ok=True)

            save_file({"url": url}, sub_output_dir / "input.json")
            save_file(html, sub_output_dir / "page.html")

            unstructured_elements = partition_html(text=html)
            save_file(
                unstructured_elements, f"{sub_output_dir}/unstructured_elements.json"
            )

            doc_markdown = convert_html_to_markdown(html, ignore_links=False)
            save_file(doc_markdown, f"{sub_output_dir}/page.md")
            doc_analysis = analyze_markdown(doc_markdown)
            save_file(doc_analysis, f"{sub_output_dir}/analysis.json")
            doc_markdown_tokens = base_parse_markdown(doc_markdown)
            save_file(doc_markdown_tokens, f"{sub_output_dir}/markdown_tokens.json")
            original_docs = derive_by_header_hierarchy(doc_markdown, ignore_links=False)
            save_file(original_docs, f"{sub_output_dir}/docs.json")

            # links = scrape_links(html, base_url=url)
            links, _ = extract_markdown_links(
                doc_markdown, base_url=url, ignore_links=True
            )
            save_file(links, sub_output_dir / "links.json")

            if result["screenshot"]:
                save_file(result["screenshot"], sub_output_dir / "screenshot.png")
                logger.success(f"Scraped {result['url']}, links count: {len(links)}")
            else:
                logger.success(
                    f"Scraped {result['url']}, links count: {len(links)}, screenshot: not taken"
                )

            html_list.append(html)
    duration = time.perf_counter() - start
    logger.info(f"Done sync scraped {len(html_list)} htmls in {duration:.2f} seconds")


if __name__ == "__main__":
    urls = [
        # "https://news.microsoft.com/source/features/ai/6-ai-trends-youll-see-more-of-in-2025",
        # "https://www.morganstanley.com/insights/articles/ai-trends-reasoning-frontier-models-2025-tmt",
        # "https://winbuzzer.com/2024/02/14/windows-10-how-to-find-and-clear-the-all-recent-files-list-xcxwbt",
        # "https://cloud.google.com/blog/topics/public-sector/5-ai-trends-shaping-the-future-of-the-public-sector-in-2025",
        # "https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-top-trends-in-tech"
        # --- MissAV links --- #
        # "https://missav.ws/dm223/en",
        # "https://missav.ws/en/aed-137",
        # "https://missav.ws/dm13/en/oksn-090",
        "https://missav.ws/dm68/en/juc-743",
    ]

    logger.info("Running sync example...")
    sync_example(urls)

    # import asyncio
    # logger.info("Running async example...")
    # asyncio.run(async_example(urls))
