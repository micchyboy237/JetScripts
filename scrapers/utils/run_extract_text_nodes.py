import os
import shutil
from jet.code.html_utils import convert_dl_blocks_to_md, preprocess_html
from jet.file.utils import load_file, save_file
from jet.scrapers.text_nodes import extract_text_nodes

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_rag_context_engineering_tips_2025_reddit/https_www_reddit_com_r_rag_comments_1mvzwrq_context_engineering_for_advanced_rag_curious_how/page.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    html_str = convert_dl_blocks_to_md(html_str)
    html_str = preprocess_html(html_str, excludes=["head", "nav", "footer", "script", "style", "button"])
    save_file(html_str, f"{output_dir}/page.html")

    text_nodes = extract_text_nodes(html_str)
    save_file(text_nodes, f"{output_dir}/text_nodes.json")

    text_elements = [dict(node.get_element_details()) for node in text_nodes]
    save_file(text_elements, f"{output_dir}/text_elements.json")
