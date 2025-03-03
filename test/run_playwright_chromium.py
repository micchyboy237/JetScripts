from jet.scrapers.browser.playwright import scrape_async_limited
import asyncio
import os
from typing import List, Optional
from jet.scrapers.browser.playwright import scrape_sync, scrape_async, PageContent, setup_sync_browser_session
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard

GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)


def build_linkedin_url(query: str, days: int, page: int) -> str:
    return (f"https://www.linkedin.com/jobs/search/?geoId=103121230&f_WT=2&"
            f"origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&keywords={query}"
            f"&f_TPR=r{days * 86400}&start={page * 25}")


def scrape_linkedin_jobs(query: str, days: int, page: int, async_mode: bool = False, wait_for_css: Optional[List[str]] = None) -> PageContent:
    url = build_linkedin_url(query, days, page)
    logger.newline()
    logger.log("URL:", url, colors=["GRAY", "ORANGE"])
    return scrape_html(url, wait_for_css=wait_for_css, async_mode=async_mode)


# if __name__ == "__main__":
#     query = "software developer"
#     days = 7
#     page = 0
#     # wait_for_css = [".results-context-header__job-count"]
#     wait_for_css = []

#     url = build_linkedin_url(query, days, page)

#     # Synchronous Example
#     browser = setup_sync_browser_session()
#     result = scrape_sync(url, wait_for_css=wait_for_css, browser=browser)
#     logger.info("Sync scrape results:")
#     logger.success({
#         "dimensions": result['dimensions'],
#         "screenshot": result['screenshot'],
#         "html": result['html'][:100],
#     })

#     # Asynchronous Example
#     async def amain():
#         result = await scrape_async(url, wait_for_css=wait_for_css)
#         return result
#     result = asyncio.run(amain())
#     logger.info("Async scrape results:")
#     logger.success({
#         "dimensions": result['dimensions'],
#         "screenshot": result['screenshot'],
#         "html": result['html'][:100],
#     })

# Example Usage


if __name__ == "__main__":
    url1 = build_linkedin_url("React Native", 14, 0)
    url2 = build_linkedin_url("React Native", 14, 1)
    url3 = build_linkedin_url("React", 1, 0)
    url4 = build_linkedin_url("Node.js", 1, 0)

    urls_to_scrape = [
        url1,
        url2,
        url3,
        url4
    ]

    max_concurrent_tasks = 4
    max_pages = 2

    results = asyncio.run(scrape_async_limited(
        urls_to_scrape, max_concurrent_tasks=max_concurrent_tasks, max_pages=max_pages))

    logger.debug(f"Results ({len(results)}):")
    copy_to_clipboard(format_json(results))
    logger.success(format_json(results))
