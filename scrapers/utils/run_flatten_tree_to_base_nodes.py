import os
import shutil
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_tree_with_text, flatten_tree_to_base_nodes

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_3/top_rag_strategies_reddit_2025/pages/medium_com_aa779_rag_in_2025_7_proven_strategies_to_deploy_retrieval_augmented_generation_at_scale_d1f71dfbfbba/page.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    save_file(html_str, f"{output_dir}/page.html")

    # Get the tree-like structure
    tree_elements = extract_tree_with_text(html_str)
    save_file(tree_elements, f"{output_dir}/tree_elements.json")

    all_nodes = flatten_tree_to_base_nodes(tree_elements)
    save_file(all_nodes, f"{output_dir}/all_nodes.json")

    all_node_parents = [node.get_parent_node() for node in all_nodes]
    save_file(all_node_parents, f"{output_dir}/all_node_parents.json")
