import os
import shutil
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_texts_by_hierarchy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_3/top_isekai_anime_2025/pages/aniflicks_com_best_isekai_anime_series_2025_top_picks_for_epic_adventures/page.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    save_file(html_str, f"{output_dir}/page.html")

    headings = extract_texts_by_hierarchy(html_str, ignore_links=True)
    save_file(headings, f"{output_dir}/headings.json")

    headings_html_strings = [node.to_html() for node in headings]
    save_file(headings_html_strings,
              f"{output_dir}/headings_html_strings.json")

    heading_parents = [node.get_parent_node() for node in headings]
    save_file(heading_parents, f"{output_dir}/heading_parents.json")

    tags_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    custom_headings = extract_texts_by_hierarchy(
        html_str, ignore_links=True, tags_to_split_on=tags_to_split_on)
    save_file(custom_headings, f"{output_dir}/custom_headings.json")

    custom_headings_html_strings = [node.to_html() for node in custom_headings]
    save_file(custom_headings_html_strings,
              f"{output_dir}/custom_headings_html_strings.json")
