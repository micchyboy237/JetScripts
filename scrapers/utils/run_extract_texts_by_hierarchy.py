import os
import shutil
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_analyzer import analyze_markdown
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_texts_by_hierarchy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
# shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_5/top_isekai_anime_2025/pages/gamerant_com_new_isekai_anime_2025/page_preprocessed.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    save_file(html_str, f"{output_dir}/page.html")

    md_content = convert_html_to_markdown(html_str)
    save_file(md_content, f"{output_dir}/md_content.md")

    analysis = analyze_markdown(md_content)
    save_file(analysis, f"{output_dir}/analysis.json")

    headings = extract_texts_by_hierarchy(html_str, ignore_links=True)
    save_file(headings, f"{output_dir}/headings.json")

    # headings = derive_by_header_hierarchy(md_content, ignore_links=True)
    # save_file(headings, f"{output_dir}/headings.json")

    # headings_html_strings = [node.html for node in headings]
    # save_file(headings_html_strings,
    #           f"{output_dir}/headings_html_strings.json")

    # heading_parents = [node.get_parent_node() for node in headings]
    # save_file(heading_parents, f"{output_dir}/heading_parents.json")

    # tags_to_split_on = [
    #     ("#", "h1"),
    #     ("##", "h2"),
    #     ("###", "h3"),
    # ]
    # custom_headings = extract_texts_by_hierarchy(
    #     html_str, ignore_links=True, tags_to_split_on=tags_to_split_on)
    # save_file(custom_headings, f"{output_dir}/custom_headings.json")

    # custom_headings_html_strings = [node.html for node in custom_headings]
    # save_file(custom_headings_html_strings,
    #           f"{output_dir}/custom_headings_html_strings.json")
