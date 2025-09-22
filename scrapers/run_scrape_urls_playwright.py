import os
import shutil
import asyncio
import time
from typing import List, Literal, Optional, TypedDict
from jet.utils.text import format_sub_dir
from jet.logger import logger
from jet.scrapers.utils import scrape_links
from jet.scrapers.playwright_utils import scrape_urls, scrape_urls_sync

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

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
        wait_for_js=True,
        use_cache=False
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
    sub_dir = f"{OUTPUT_DIR}/sync_results"

    screenshots_dir = os.path.join(sub_dir, "screenshots")
    html_files_dir = os.path.join(sub_dir, "html_files")
    os.makedirs(screenshots_dir, exist_ok=True)
    os.makedirs(html_files_dir, exist_ok=True)

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
        wait_for_js=True,
        use_cache=False
    )

    for result in results:
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
    logger.info(f"Done sync scraped {len(html_list)} htmls in {duration:.2f} seconds")


if __name__ == "__main__":
    urls = [
        "https://www.asfcxcvqawe.com",
        "https://www.imdb.com/list/ls505070747",
        "https://myanimelist.net/stacks/32507",
        "https://example.com",
        "https://python.org",
        "https://github.com",
        "https://httpbin.org/html",
        "https://www.wikipedia.org/",
        "https://www.mozilla.org",
        "https://www.stackoverflow.com",
    ]

    logger.info("Running sync example...")
    sync_example(urls)

    logger.info("Running async example...")
    asyncio.run(async_example(urls))
