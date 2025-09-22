import os
import shutil
import re
import asyncio
import platform
import sys
import base64
from typing import AsyncIterator, List, Literal, Optional, Tuple
from playwright.async_api import async_playwright, BrowserContext
from fake_useragent import UserAgent
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.utils import scrape_links
from jet.scrapers.playwright_utils import scrape_urls, scrape_urls_sync
from tqdm.asyncio import tqdm_asyncio

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(output_dir, ignore_errors=True)

screenshots_dir = os.path.join(output_dir, "screenshots")
html_files_dir = os.path.join(output_dir, "html_files")
os.makedirs(screenshots_dir, exist_ok=True)
os.makedirs(html_files_dir, exist_ok=True)

async def amain(urls):
    html_list = []

    async for url, status, html, screenshot in scrape_urls(
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
        if status == "completed":
            if not html:
                continue
            all_links = scrape_links(html, base_url=url)
            safe_filename = re.sub(r'[^\w\-_\.]', '_', url)
            if screenshot:
                screenshot_path = os.path.join(screenshots_dir, f"{safe_filename}.png")
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot)
                logger.success(f"Scraped {url}, links count: {len(all_links)}, screenshot saved: {screenshot_path}")
            else:
                logger.success(f"Scraped {url}, links count: {len(all_links)}, screenshot: not taken")
            html_path = os.path.join(html_files_dir, f"{safe_filename}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.success(f"Saved HTML for {url} to: {html_path}")
            html_list.append(html)
    logger.info(f"Done async scraped {len(html_list)} htmls")

def main(urls):
    html_list = []

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

    for url, status, html, screenshot in results:
        if status == "completed":
            if not html:
                continue
            all_links = scrape_links(html, base_url=url)
            safe_filename = re.sub(r'[^\w\-_\.]', '_', url)
            if screenshot:
                screenshot_path = os.path.join(screenshots_dir, f"{safe_filename}.png")
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot)
                logger.success(f"Scraped {url}, links count: {len(all_links)}, screenshot saved: {screenshot_path}")
            else:
                logger.success(f"Scraped {url}, links count: {len(all_links)}, screenshot: not taken")
            html_path = os.path.join(html_files_dir, f"{safe_filename}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.success(f"Saved HTML for {url} to: {html_path}")
            html_list.append(html)
    logger.info(f"Done sync scraped {len(html_list)} htmls")


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

    main(urls)
    # asyncio.run(amain(urls))
