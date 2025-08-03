import os
import shutil
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_by_heading_hierarchy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/pages/www_ranker_com_list_best_isekai_anime_2025_anna_lindwasser/page.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    save_file(html_str, f"{output_dir}/page.html")

    headings = extract_by_heading_hierarchy(html_str)
    save_file(headings, f"{output_dir}/headings.json")

    headings_html_strings = [node.html for node in headings]
    save_file(headings_html_strings,
              f"{output_dir}/headings_html_strings.json")

    heading_parents = [node.get_parent_node() for node in headings]
    save_file(heading_parents, f"{output_dir}/heading_parents.json")
