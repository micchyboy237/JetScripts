import os
import shutil
from jet.code.html_utils import convert_dl_blocks_to_md, preprocess_html
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_tree_with_text, flatten_tree_to_base_nodes

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

    # Get the tree-like structure
    tree_elements = extract_tree_with_text(html_str)
    save_file(tree_elements, f"{output_dir}/tree_elements.json")

    all_nodes = flatten_tree_to_base_nodes(tree_elements)
    save_file(all_nodes, f"{output_dir}/all_nodes.json")

    all_links = [node.get_links() for node in all_nodes]
    save_file(all_links, f"{output_dir}/all_links.json")

    all_html = [node.get_html() for node in all_nodes]
    save_file(all_html, f"{output_dir}/all_html.json")

    all_elements = [dict(node.get_element_details()) for node in all_nodes]
    save_file(all_elements, f"{output_dir}/all_elements.json")
    
    all_clickable_elements = [dict(node.get_element_details()) for node in all_nodes if node.is_clickable]
    save_file(all_clickable_elements, f"{output_dir}/all_clickable_elements.json")
