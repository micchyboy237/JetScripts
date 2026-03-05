import shutil
from pathlib import Path

from jet.file.utils import load_file, save_file
from jet.scrapers.utils import (
    get_flattened_parents_with_most_children,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_scrape_urls_playwright/missav_ws_dm223_en/sync_results/page.html"
    html = load_file(html_file)

    flattened_parents_with_most_children = get_flattened_parents_with_most_children(
        html
    )
    save_file(
        flattened_parents_with_most_children,
        OUTPUT_DIR / "flattened_parents_with_most_children.json",
    )
