import os
import shutil

from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_title_and_metadata, scrape_links

# Example usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0],
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    html_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_playwright_scrape_urls/deepeval_com_docs_metrics_introduction/sync_results/page.html"
    html_str: str = load_file(html_path)

    # Run and save results without base URL
    all_links_no_base = scrape_links(html_str, base=None)
    save_file(all_links_no_base, os.path.join(output_dir, "all_links.no_base.json"))

    # Run and save results with base URL
    base_url = "https://deepeval.com"
    all_links_with_base = scrape_links(html_str, base=base_url)
    save_file(all_links_with_base, os.path.join(output_dir, "all_links.with_base.json"))

    # Save title and metadata (same file for both runs)
    title_and_metadata = extract_title_and_metadata(html_str)
    save_file(title_and_metadata, os.path.join(output_dir, "title_and_metadata.json"))
