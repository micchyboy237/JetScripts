import os
import shutil
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_tree_with_text, get_leaf_nodes

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    save_file(html_str, f"{output_dir}/page.html")

    # Get the tree-like structure
    tree_elements = extract_tree_with_text(html_str)
    save_file(tree_elements, f"{output_dir}/tree_elements.json")

    leaf_nodes = get_leaf_nodes(tree_elements)
    save_file(leaf_nodes, f"{output_dir}/leaf_nodes.json")

    leaf_node_parents = [node.get_parent_node() for node in leaf_nodes]
    save_file(leaf_node_parents, f"{output_dir}/leaf_node_parents.json")
