import os
import shutil
from typing import Optional

from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.file.utils import save_file
from jet.logger import logger
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import safe_path_from_url, search_data
from jet.utils.url_utils import normalize_url


def get_url_html_tuples(urls: list[str], top_n: int = 3, num_parallel: int = 3, min_header_count: int = 10, min_avg_word_count: int = 10, output_dir: Optional[str] = None) -> list[tuple[str, str]]:
    urls = [normalize_url(url) for url in urls]

    if output_dir:
        sub_dir = os.path.join(output_dir, "searched_html")
        shutil.rmtree(sub_dir, ignore_errors=True)

    url_html_tuples = []
    for url, html in scrape_multiple_urls(urls, top_n=top_n, num_parallel=num_parallel, min_header_count=min_header_count, min_avg_word_count=min_avg_word_count):
        url_html_tuples.append((url, html))

        if output_dir:
            output_dir_url = safe_path_from_url(url, sub_dir)
            os.makedirs(output_dir_url, exist_ok=True)

            md_text = html_to_markdown(html)
            headers = get_md_header_contents(md_text)
            header_texts = [header["content"] for header in headers]

            save_file(html, os.path.join(output_dir_url, "doc.html"))
            save_file("\n\n".join(header_texts),
                      os.path.join(output_dir_url, "doc.md"))

    logger.success(f"Done scraping urls {len(url_html_tuples)}")
    return url_html_tuples


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "generated",
                              os.path.splitext(os.path.basename(__file__))[0])

    query = "List trending isekai anime today."

    search_results = search_data(query)
    save_file(search_results, os.path.join(
        output_dir, "search_results.json"))

    urls = [item["url"] for item in search_results]
    url_html_tuples = get_url_html_tuples(urls, output_dir=output_dir)
