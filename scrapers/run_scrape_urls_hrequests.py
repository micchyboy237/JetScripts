import asyncio

from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.logger import logger
from jet.scrapers.hrequests_utils import scrape_urls
from jet.scrapers.utils import scrape_links


if __name__ == "__main__":
    urls = [
        "https://www.imdb.com/list/ls505070747",
        "https://myanimelist.net/stacks/32507",
        "https://example.com",
        "https://python.org",
        "https://github.com",
        "https://httpbin.org/html",
        "https://www.wikipedia.org/",
        "https://www.mozilla.org",
        "https://www.stackoverflow.com"
    ]
    results = asyncio.run(scrape_urls(urls, num_parallel=3))
    for url, html_str in zip(urls, results):
        if html_str:
            all_links = scrape_links(html_str, base_url=url)
            headers = get_md_header_contents(html_str)
            logger.success(
                f"Scraped {url}, headers length: {len(headers)}, links count: {len(all_links)}")
        else:
            logger.error(f"Failed to fetch {url}")
