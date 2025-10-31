import os
import shutil
from jet.code.html_utils import convert_dl_blocks_to_md
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_tree_with_text, get_parents_with_common_class

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    html_str = convert_dl_blocks_to_md(html_str)
    save_file(html_str, f"{output_dir}/page.html")

    # Get the tree-like structure
    tree_elements = extract_tree_with_text(html_str)
    save_file(tree_elements, f"{output_dir}/tree_elements.json")

    parents_with_common_tags = get_parents_with_common_class(
        tree_elements, match_type="tag")
    save_file(parents_with_common_tags,
              f"{output_dir}/parents_with_common_tags.json")

    parents_with_common_classes = get_parents_with_common_class(
        tree_elements, match_type="class")
    save_file(parents_with_common_classes,
              f"{output_dir}/parents_with_common_classes.json")

    parents_with_common_tags_and_classes = get_parents_with_common_class(
        tree_elements, match_type="both")
    save_file(parents_with_common_tags_and_classes,
              f"{output_dir}/parents_with_common_tags_and_classes.json")

    all_contents = [node.get_content()
                    for node in parents_with_common_tags_and_classes]
    save_file(all_contents, f"{output_dir}/all_contents.json")

    all_headers = [node.get_header()
                   for node in parents_with_common_tags_and_classes
                   if node.get_header()]
    save_file(all_headers, f"{output_dir}/all_headers.json")

    all_html = [node.get_html()
                for node in parents_with_common_tags_and_classes]
    save_file(all_html, f"{output_dir}/all_html.json")
