import os
import shutil
from jet.scrapers.crawler.web_crawler import WebCrawler, sort_urls_numerically
from jet.file.utils import load_file, save_data, save_file
from jet.logger import logger
from jet.scrapers.utils import extract_title_and_metadata, scrape_links
from jet.transformers.formatters import format_json

# Example usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    html_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/_temp_test/searched_html_gamerant_com_new-isekai-anime-2024-upcoming/doc.html"
    html_str: str = load_file(html_path)
    url = "https://gamerant.com/new-isekai-anime-2024-upcoming"

    all_links = scrape_links(html_str, base_url=url)
    save_file(all_links, os.path.join(output_dir, "all_links.json"))

    title_and_metadata = extract_title_and_metadata(html_str)
    save_file(title_and_metadata, os.path.join(
        output_dir, "title_and_metadata.json"))
