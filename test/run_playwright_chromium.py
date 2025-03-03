import asyncio
import os
from typing import List, Optional
from jet.scrapers.browser.playwright import scrape_sync, scrape_async, ScrapeHTMLResult, setup_sync_browser_session
from jet.logger import logger

GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)


def build_linkedin_url(query: str, days: int, page: int) -> str:
    return (f"https://www.linkedin.com/jobs/search/?geoId=103121230&f_WT=2&"
            f"origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&keywords={query}"
            f"&f_TPR=r{days * 86400}&start={page * 25}")


def scrape_linkedin_jobs(query: str, days: int, page: int, async_mode: bool = False, wait_for_css: Optional[List[str]] = None) -> ScrapeHTMLResult:
    url = build_linkedin_url(query, days, page)
    logger.newline()
    logger.log("URL:", url, colors=["GRAY", "ORANGE"])
    return scrape_html(url, wait_for_css=wait_for_css, async_mode=async_mode)


if __name__ == "__main__":
    query = "software developer"
    days = 7
    page = 0
    # wait_for_css = [".results-context-header__job-count"]
    wait_for_css = []

    url = build_linkedin_url(query, days, page)

    # Synchronous Example
    browser = setup_sync_browser_session()
    result = scrape_sync(url, wait_for_css=wait_for_css, browser=browser)
    logger.info("Sync scrape results:")
    logger.success({
        "dimensions": result['dimensions'],
        "screenshot": result['screenshot'],
        "html": result['html'][:100],
    })

    # Asynchronous Example
    async def amain():
        result = await scrape_async(url, wait_for_css=wait_for_css)
        return result
    result = asyncio.run(amain())
    logger.info("Async scrape results:")
    logger.success({
        "dimensions": result['dimensions'],
        "screenshot": result['screenshot'],
        "html": result['html'][:100],
    })
