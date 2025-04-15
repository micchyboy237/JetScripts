import asyncio
from playwright.async_api import async_playwright
from typing import List


async def fetch_page_content(page, url: str) -> str:
    try:
        await page.goto(url, timeout=15000)
        await page.wait_for_load_state("networkidle")
        content = await page.content()
        return content
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return ""


async def scrape_with_playwright(urls: List[str]) -> List[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        pages = [await context.new_page() for _ in urls]
        tasks = [fetch_page_content(page, url)
                 for page, url in zip(pages, urls)]
        results = await asyncio.gather(*tasks)

        await browser.close()
        return results


def scrape_multiple_urls(urls: List[str]) -> List[str]:
    return asyncio.run(scrape_with_playwright(urls))


if __name__ == "__main__":
    urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://www.python.org"
    ]
    html_list = scrape_multiple_urls(urls)
    for i, html in enumerate(html_list):
        print(f"--- HTML from {urls[i]} ---\n{html[:300]}...\n")
