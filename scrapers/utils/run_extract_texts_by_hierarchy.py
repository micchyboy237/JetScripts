import os
import shutil
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_texts_by_hierarchy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_3/top_isekai_anime_2025/pages/gamerant_com_new_isekai_anime_2025/page.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    save_file(html_str, f"{output_dir}/page.html")

    headings = extract_texts_by_hierarchy(html_str, ignore_links=True)
    save_file(headings, f"{output_dir}/headings.json")
