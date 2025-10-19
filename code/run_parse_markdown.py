import os
import shutil

from jet.code.markdown_utils import parse_markdown
from jet.file.utils import load_file, save_file

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html")

    results_ignore_links = parse_markdown(html, ignore_links=True)
    results_with_links = parse_markdown(html, ignore_links=False)

    save_file(results_ignore_links, f"{OUTPUT_DIR}/results_ignore_links.json")
    save_file(results_with_links, f"{OUTPUT_DIR}/results_with_links.json")

    results_ignore_links = parse_markdown(
        html, merge_headers=False, merge_contents=False, ignore_links=True)
    results_with_links = parse_markdown(
        html, merge_headers=False, merge_contents=False, ignore_links=False)

    save_file(results_ignore_links,
              f"{OUTPUT_DIR}/results_no_merge_ignore_links.json")
    save_file(results_with_links,
              f"{OUTPUT_DIR}/results_no_merge_with_links.json")
