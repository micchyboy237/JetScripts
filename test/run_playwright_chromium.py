from jet.scrapers.browser.playwright_helpers import scrape_async_limited
import asyncio
import os
from typing import List, Optional
from jet.scrapers.browser.playwright_helpers import scrape_sync, scrape_async, PageContent, setup_sync_browser_session
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from shared.data_types.job import JobData

GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)


def build_linkedin_url(query: str, days: int, page: int) -> str:
    return (f"https://www.linkedin.com/jobs/search/?geoId=103121230&f_WT=2&"
            f"origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&keywords={query}"
            f"&f_TPR=r{days * 86400}&start={page * 25}")


async def async_scrape_jobs(urls_to_scrape, max_concurrent_tasks):
    return await scrape_async_limited(urls_to_scrape, max_concurrent_tasks=max_concurrent_tasks)


def parse_jobs(job_links: list[str], max_concurrent_tasks: int = 2) -> list[JobData]:
    if asyncio.get_event_loop().is_running():
        results = asyncio.create_task(async_scrape_jobs(
            job_links, max_concurrent_tasks))
        # Correctly wait for the task
        jobs_with_details = asyncio.run(results)
    else:
        jobs_with_details = asyncio.run(
            async_scrape_jobs(urls_to_scrape, max_concurrent_tasks))

    return jobs_with_details  # Return the processed job details


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

    max_concurrent_tasks = 2

    jobs_with_details = parse_jobs(
        urls_to_scrape, max_concurrent_tasks=max_concurrent_tasks)

    results = asyncio.run(scrape_async_limited(
        urls_to_scrape, max_concurrent_tasks=max_concurrent_tasks))

    logger.debug(f"Results ({len(results)}):")
    copy_to_clipboard(format_json(results))
    logger.success(format_json(results))
