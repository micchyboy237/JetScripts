import os
import shutil
from typing import List
from jet.code.html_utils import convert_dl_blocks_to_md, preprocess_html
from jet.file.utils import load_file, save_file
from jet.scrapers.text_nodes import extract_text_nodes, BaseNode

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main() -> None:
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_rag_context_engineering_tips_2025_reddit/https_www_reddit_com_r_rag_comments_1mvzwrq_context_engineering_for_advanced_rag_curious_how/page.html"

    html_str: str = load_file(html_file)
    html_str = convert_dl_blocks_to_md(html_str)
    html_str = preprocess_html(html_str, excludes=["head", "nav", "footer", "script", "style", "button"])
    save_file(html_str, f"{OUTPUT_DIR}/page.html")

    # Extract text nodes with default excludes and timeout
    nodes: List[BaseNode] = extract_text_nodes(
        source=html_str,
        excludes=["nav", "footer", "script", "style"],
        timeout_ms=1000
    )

    save_file(nodes, f"{OUTPUT_DIR}/nodes.json")


if __name__ == "__main__":
    main()
