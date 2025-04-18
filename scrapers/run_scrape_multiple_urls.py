import asyncio

from jet.scrapers.browser.playwright_utils import scrape_multiple_urls


if __name__ == "__main__":
    urls = [
        "https://www.ranker.com/list/best-reincarnation-anime/ranker-anime",
        "https://fictionhorizon.com/best-reincarnation-anime",
        "https://otakusnotes.com/best-reincarnation-anime-of-all-time"
    ]
    # Run the async function
    html_list = asyncio.run(scrape_multiple_urls(urls))
    for i, html in enumerate(html_list):
        print(f"--- HTML from {urls[i]} ---\n{html[:300]}...\n")
