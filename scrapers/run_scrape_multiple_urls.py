import os
import sys
import time
from fake_useragent import UserAgent
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.file.utils import save_file
from jet.logger import logger
from jet.scrapers.browser.playwright_utils import ascrape_multiple_urls, scrape_multiple_urls
from jet.scrapers.utils import safe_path_from_url, validate_headers
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from typing import Any, AsyncGenerator, List, Tuple, Generator

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


async def main():
    top_n = 3
    num_parallel = 3
    min_header_count = 5
    sample_urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://www.wikipedia.org/",
        "https://www.bbc.com/news",
        "https://www.cnn.com",
        "https://www.nytimes.com",
        "https://www.mozilla.org",
        "https://www.stackoverflow.com",
        "https://news.ycombinator.com",
        "https://www.reddit.com"
    ]

    print("\nStarting sync scrape...")
    url_html_tuples = []
    for url, content in scrape_multiple_urls(sample_urls, top_n=top_n, num_parallel=num_parallel, min_header_count=min_header_count):
        logger.success(f"Scraped {url}, content length: {len(content)}")
        url_html_tuples.append((url, content))
    logger.orange(f"Done sync scrape {len(url_html_tuples)}")
    logger.newline()

    for url, html in url_html_tuples:
        sub_dir = os.path.join(OUTPUT_DIR, "sync_searched_html")
        output_dir_url = safe_path_from_url(url, sub_dir)
        os.makedirs(output_dir_url, exist_ok=True)

        save_file(html, os.path.join(output_dir_url, "doc.html"))
        save_file("\n\n".join([header["content"] for header in get_md_header_contents(
            html)]), os.path.join(output_dir_url, "doc.md"))

    print("Starting async scrape...")
    aurl_html_tuples = []
    async for url, content in ascrape_multiple_urls(sample_urls, top_n=top_n, num_parallel=num_parallel, min_header_count=min_header_count):
        logger.success(f"Scraped {url}, content length: {len(content)}")
        aurl_html_tuples.append((url, content))
    logger.orange(f"Done async scrape {len(aurl_html_tuples)}")
    logger.newline()

    for url, html in aurl_html_tuples:
        sub_dir = os.path.join(OUTPUT_DIR, "async_searched_html")
        output_dir_url = safe_path_from_url(url, sub_dir)
        os.makedirs(output_dir_url, exist_ok=True)

        save_file(html, os.path.join(output_dir_url, "doc.html"))
        save_file("\n\n".join([header["content"] for header in get_md_header_contents(
            html)]), os.path.join(output_dir_url, "doc.md"))

try:
    asyncio.run(main())
except KeyboardInterrupt:
    logger.warning("Scraping interrupted by user.")
    sys.exit(0)
