import os
import shutil
from typing import List
from jet.file.utils import load_file, save_file
from jet.scrapers.header_hierarchy import extract_header_hierarchy, HtmlHeaderDoc

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main() -> None:
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_scrape_urls_playwright/async_results/html_files/https_cloud_google_com_blog_topics_public_sector_5_ai_trends_shaping_the_future_of_the_public_sector_in_2025.html"
    html_str: str = load_file(html_file)

    save_file(html_str, f"{OUTPUT_DIR}/page.html")

    # Extract header hierarchy with default excludes and timeout
    headings: List[HtmlHeaderDoc] = extract_header_hierarchy(
        source=html_str,
        excludes=["nav", "footer", "script", "style"],
        timeout_ms=1000
    )

    save_file(headings, f"{OUTPUT_DIR}/headings.json")


if __name__ == "__main__":
    main()
