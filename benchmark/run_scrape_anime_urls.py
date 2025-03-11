import json
from typing import Optional, TypedDict
from jet.code.splitter_markdown_utils import extract_html_header_contents
from jet.file.utils import load_file
from jet.logger import logger
from jet.scrapers.browser.playwright import scrape_async_limited, setup_sync_browser_session
from jet.search.searxng import search_searxng, SearchResult
import requests


class BaseScraper:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self.browser = setup_sync_browser_session(headless=False)
        self.session = self.browser.new_page()

    async def get_data(self):
        results = await self.scrape_urls(self.urls)

        data = {}
        for result in results:
            html_str = result["html"]

            header_contents = extract_html_header_contents(html_str)

    async def scrape_urls(self, urls: list[str]):
        results = await scrape_async_limited(
            urls=urls,
            max_concurrent_tasks=2,  # More concurrent tasks
            headless=True
        )
        return results


def search_data(query) -> list[SearchResult]:
    filter_sites = [
        # "https://easypc.com.ph",
        # "9anime",
        # "zoro"
        "aniwatch"
    ]
    engines = [
        "google",
        "brave",
        "duckduckgo",
        "bing",
        "yahoo",
    ]
    results: list[SearchResult] = search_searxng(
        query_url="http://searxng.local:8080/search",
        query=query,
        min_score=0.2,
        filter_sites=filter_sites,
        engines=engines,
        config={
            "port": 3101
        },
    )

    return results


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/benchmark/data/aniwatch_history.json"
    titles: list[str] = load_file(data_file) or []

    data = []
    for title in titles:
        query = f"How many seasons and episodes does \"{title}\" anime have?"

        results = search_data(query)

        urls = [d["url"] for d in results]
