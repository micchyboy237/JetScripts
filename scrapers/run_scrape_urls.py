from jet.logger import logger
from jet.scrapers.selenium import UrlScraper
from jet.scrapers.preprocessor import scrape_markdown
from jet.scrapers.hrequests import request_url
from jet.transformers import to_snake_case
import os
import hashlib
import json

# Set working directory to script location
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)

output_dir = os.path.join(file_dir, "generated")
os.makedirs(output_dir, exist_ok=True)

cache_dir = os.path.join(file_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)


def cache_file_path(url: str) -> str:
    # Use URL hash for cache filename
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, f"{url_hash}.json")


def load_cache(url: str) -> dict:
    cache_path = cache_file_path(url)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(url: str, data: dict):
    cache_path = cache_file_path(url)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main_selenium(url):
    url_scraper = UrlScraper()
    html_str = url_scraper.scrape_url(url)
    return html_str


def main_hrequests(url):
    html_parser = request_url(url, showBrowser=True)
    html_str = html_parser.raw_html.decode('utf-8')
    return html_str


def scrape_url(url: str) -> dict:
    # Check if the data is cached
    cached_result = load_cache(url)
    if cached_result:
        logger.log("CACHE", "Cache hit for:", url, colors=[
                   "GRAY", "SUCCESS", "BRIGHT_SUCCESS"])
        return cached_result

    html_str = main_selenium(url)
    md_result = scrape_markdown(html_str)
    md_str = md_result['content']

    result = {
        "html": html_str,
        "markdown": md_str,
        "title": md_result["title"],
        "headings": md_result["headings"],
    }

    # Save to cache
    save_cache(url, result)

    return result


def generate_filename(url: str, extension: str) -> str:
    # Convert URL to snake case
    snake_case_url = to_snake_case(url)
    return os.path.join(output_dir, f"{snake_case_url}{extension}")


def scrape_urls(urls: list):
    for url in urls:
        result = scrape_url(url)

        html_output_file = generate_filename(url, ".html")
        md_output_file = generate_filename(url, ".md")

        with open(html_output_file, "w", encoding="utf-8") as f:
            f.write(result["html"])
        with open(md_output_file, "w", encoding="utf-8") as f:
            f.write(result["markdown"])

        logger.log("HTML", f"({len(result['html'])})", "saved to:", html_output_file,
                   colors=["GRAY", "SUCCESS", "GRAY", "BRIGHT_SUCCESS"])
        logger.log("MD", f"({len(result['markdown'])})", "saved to:", md_output_file,
                   colors=["GRAY", "SUCCESS", "GRAY", "BRIGHT_SUCCESS"])


if __name__ == "__main__":
    urls = [
        "https://www.imdb.com/title/tt32812118/",
        "https://9animetv.to/watch/ill-become-a-villainess-who-goes-down-in-history-19334?ep=129043",
        "https://www.crunchyroll.com/series/GQWH0M17X/ill-become-a-villainess-who-goes-down-in-history",
        "https://zorotv.pro.in/ill-become-a-villainess-who-goes-down-in-history-episode-4/",
    ]

    scrape_urls(urls)
