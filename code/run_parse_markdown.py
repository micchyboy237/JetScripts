import os
import shutil

from jet.code.html_utils import convert_dl_blocks_to_md
from jet.code.markdown_utils import parse_markdown
from jet.file.utils import load_file, save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_scrape_urls_playwright/missav_ws_dm223_en/sync_results/page.html"
    html = load_file(html_file)
    html = convert_dl_blocks_to_md(html)

    results_ignore_links = parse_markdown(html, ignore_links=True)
    results_with_links = parse_markdown(html, ignore_links=False)

    save_file(results_ignore_links, f"{OUTPUT_DIR}/results_ignore_links.json")
    save_file(results_with_links, f"{OUTPUT_DIR}/results_with_links.json")

    results_ignore_links = parse_markdown(
        html, merge_headers=False, merge_contents=False, ignore_links=True
    )
    results_with_links = parse_markdown(
        html, merge_headers=False, merge_contents=False, ignore_links=False
    )

    save_file(results_ignore_links, f"{OUTPUT_DIR}/results_no_merge_ignore_links.json")
    save_file(results_with_links, f"{OUTPUT_DIR}/results_no_merge_with_links.json")
