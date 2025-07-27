import os
import shutil
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_tree_with_text, get_significant_nodes

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

    significant_nodes = get_significant_nodes(tree_elements)
    save_file(significant_nodes, f"{output_dir}/significant_nodes.json")

    significant_node_parents = [node.get_parent_node()
                                for node in significant_nodes]
    save_file(significant_node_parents,
              f"{output_dir}/significant_node_parents.json")

    for num, significant_node in enumerate(significant_nodes, start=1):
        sub_output_dir = f"{output_dir}/significant_node_{num}"
        html = significant_node.html

        save_file(html, f"{sub_output_dir}/node.html")

        md_content = convert_html_to_markdown(html)
        save_file(md_content, f"{sub_output_dir}/node.md")

        analysis = analyze_markdown(md_content)
        save_file(analysis, f"{sub_output_dir}/analysis.json")

        markdown_tokens = base_parse_markdown(md_content)
        save_file({
            "count": len(markdown_tokens),
            "tokens": markdown_tokens,
        }, f"{sub_output_dir}/markdown_tokens.json")

        header_docs = derive_by_header_hierarchy(md_content)
        save_file({
            "count": len(header_docs),
            "documents": header_docs,
        }, f"{sub_output_dir}/header_docs.json")
