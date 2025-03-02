import asyncio
import os
from typing import TypedDict, List, Optional
from jet.scrapers.browser.playwright import scrape_async, scrape_sync
from jet.logger import logger

GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)


class PageDimensions(TypedDict):
    width: int
    height: int
    deviceScaleFactor: float


def build_linkedin_url(query: str, days: int, page: int) -> str:
    return (f"https://www.linkedin.com/jobs/search/?geoId=103121230&f_WT=2&"
            f"origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&keywords={query}"
            f"&f_TPR=r{days * 86400}&start={page * 25}")


def scrape_linkedin_jobs(query: str, days: int, page: int, async_mode: bool = False, wait_for_css: Optional[List[str]] = None) -> str:
    url = build_linkedin_url(query, days, page)
    logger.newline()
    logger.log("URL:", url, colors=["GRAY", "ORANGE"])
    return asyncio.run(scrape_async(url, wait_for_css)) if async_mode else scrape_sync(url, wait_for_css)


if __name__ == "__main__":
    query = "software developer"
    days = 7
    page = 0
    wait_for_css = [".results-context-header__job-count"]
    html_result = scrape_linkedin_jobs(
        query, days, page, async_mode=False, wait_for_css=wait_for_css)
    print("HTML Content:", html_result[:500])

    html_result = scrape_linkedin_jobs(
        query, days, page, async_mode=True, wait_for_css=wait_for_css)
    print("HTML Content:", html_result[:500])
