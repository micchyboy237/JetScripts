import os
import shutil

from jet.code.markdown_utils import base_analyze_markdown
from jet.file.utils import load_file, save_file

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

md_content = """
Release Date
 2025 - 2025-00-00
"""

if __name__ == "__main__":
    html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html")

    results = base_analyze_markdown(md_content)



    save_file(results, f"{OUTPUT_DIR}/results.json")
