import os
import shutil

from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import load_file, save_file

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    # html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/node_extraction/sample.html")
    # html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html")
    html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_rag_context_engineering_tips_2025_reddit/https_www_reddit_com_r_rag_comments_1mvzwrq_context_engineering_for_advanced_rag_curious_how/page.html")

    results_ignore_links = derive_by_header_hierarchy(html, ignore_links=True, valid_sentences_only=False)
    save_file(results_ignore_links, f"{OUTPUT_DIR}/results_ignore_links.json")

    results_with_links = derive_by_header_hierarchy(html, ignore_links=False, valid_sentences_only=False)
    save_file(results_with_links, f"{OUTPUT_DIR}/results_with_links.json")
