import os
import shutil
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_by_heading_hierarchy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"
    html_str: str = load_file(html_file)
    save_file(html_str, f"{OUTPUT_DIR}/page.html")

    headings = extract_by_heading_hierarchy(html_str)
    save_file(headings, f"{OUTPUT_DIR}/headings.json")
