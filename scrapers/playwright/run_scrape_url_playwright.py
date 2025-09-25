import os
import shutil
import asyncio
import time
from typing import List, Literal, Optional, TypedDict
# from jet.code.splitter_markdown_utils_old import get_md_header_docs
from jet.file.utils import save_file
from jet.utils.text import format_sub_dir
from jet.logger import logger
from jet.scrapers.utils import scrape_links, clean_text
from jet.scrapers.playwright_utils import scrape_url_sync
from jet.scrapers.preprocessor import base_convert_html_to_markdown, html_to_markdown
# from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, parse_markdown, derive_by_header_hierarchy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.orange(f"Log file: {log_file}")


class ScrapeResult(TypedDict):
    url: str
    status: Literal["started", "completed", "failed_no_html", "failed_error"]
    html: Optional[str]
    screenshot: Optional[bytes]


def usage_example(url: str, use_cache: bool = False) -> None:
    html_list = []
    all_links = []

    start = time.perf_counter()
    result = scrape_url_sync(
        url,
        max_retries=3,
        with_screenshot=True,
        headless=False,
        wait_for_js=True,
        use_cache=use_cache
    )

    if result["status"] == "completed" and result["html"]:
        scraped_links = scrape_links(result["html"], base_url=result["url"])
        url_sub_dir = format_sub_dir(result["url"])
        sub_dir = os.path.join(OUTPUT_DIR, url_sub_dir)
        shutil.rmtree(sub_dir, ignore_errors=True)

        if result["screenshot"]:
            screenshot_path = os.path.join(sub_dir, "screenshot.png")
            save_file(result["screenshot"], screenshot_path)
        else:
            logger.success(
                f"Scraped {result['url']}, links count: {len(scraped_links)}, screenshot: not taken"
            )

        html_path = os.path.join(sub_dir, "page.html")
        save_file(result["html"], html_path)

        md_path = os.path.join(sub_dir, "markdown.md")
        # html_markdown = html_to_markdown(result["html"])
        html_markdown = base_convert_html_to_markdown(result["html"], ignore_links=False)
        save_file(html_markdown, md_path)

        headers_path = os.path.join(sub_dir, "headers.json")
        save_file(derive_by_header_hierarchy(html_markdown), headers_path)

        links_path = os.path.join(sub_dir, "links.json")
        save_file(scraped_links, links_path)

        html_list.append(result["html"])
        all_links.extend(scraped_links)

        save_file(all_links, f"{OUTPUT_DIR}/all_links.json")

    duration = time.perf_counter() - start
    logger.info(f"Done sync scraped {len(html_list)} htmls in {duration:.2f} seconds")


if __name__ == "__main__":
    url = "https://towardsdatascience.com/evaluating-your-rag-solution/"
    use_cache = True

    logger.info("Running sync example...")
    usage_example(url, use_cache)
