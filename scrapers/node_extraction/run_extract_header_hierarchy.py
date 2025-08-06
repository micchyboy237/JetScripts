import os
import shutil
import json
from typing import List
from jet.file.utils import load_file, save_file
from jet.scrapers.header_hierarchy import extract_header_hierarchy, HeaderDoc

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main() -> None:
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_5/top_isekai_anime_2025/pages/gamerant_com_new_isekai_anime_2025/page_preprocessed.html"
    html_str: str = load_file(html_file)

    # Extract header hierarchy with default excludes and timeout
    headings: List[HeaderDoc] = extract_header_hierarchy(
        source=html_str,
        excludes=["nav", "footer", "script", "style"],
        timeout_ms=1000
    )

    save_file(headings, f"{OUTPUT_DIR}/headings.json")


if __name__ == "__main__":
    main()
