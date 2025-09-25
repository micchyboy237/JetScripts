import os
import shutil
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_by_heading_hierarchy

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/generated/run_rag_search/news_microsoft_com_source_features_ai_6_ai_trends_youll_see_more_of_in_2025/page.html"

    output_dir = OUTPUT_DIR

    html_str: str = load_file(html_file)
    save_file(html_str, f"{output_dir}/page.html")

    header_docs = extract_by_heading_hierarchy(html_str)
    save_file(header_docs, f"{output_dir}/header_docs.json")

    md_content = "\n\n".join([node.text for node in header_docs])
    save_file(md_content, f"{output_dir}/markdown.md")

    headers = [node.header for node in header_docs]
    save_file(headers, f"{output_dir}/headers.json")

    contents = [node.content for node in header_docs]
    save_file(contents, f"{output_dir}/contents.json")

    texts = [node.text for node in header_docs]
    save_file(texts, f"{output_dir}/texts.json")

    headings_html_strings = [node.get_html() for node in header_docs]
    save_file(headings_html_strings,
              f"{output_dir}/headings_html_strings.json")

    heading_parents = [node.get_parent_node() for node in header_docs]
    save_file(heading_parents, f"{output_dir}/heading_parents.json")
