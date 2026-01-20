import os
import shutil

from jet.code.html_utils import convert_dl_blocks_to_md
from jet.code.markdown_utils import base_analyze_markdown, convert_html_to_markdown
from jet.file.utils import load_file, save_file
from jet.utils.print_utils import print_dict_types

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/node_extraction/sample.html")
    # html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_rag_context_engineering_tips_2025_reddit/https_www_reddit_com_r_rag_comments_1mvzwrq_context_engineering_for_advanced_rag_curious_how/page.html")
    html = convert_dl_blocks_to_md(html)

    md_content_with_links = convert_html_to_markdown(html, ignore_links=False)
    results_with_links = base_analyze_markdown(html, ignore_links=False)

    md_content_ignore_links = convert_html_to_markdown(html, ignore_links=True)
    results_ignore_links = base_analyze_markdown(html, ignore_links=True)

    print_dict_types(results_with_links)

    save_file(results_with_links, f"{OUTPUT_DIR}/results_with_links.json")
    save_file(results_ignore_links, f"{OUTPUT_DIR}/results_ignore_links.json")
