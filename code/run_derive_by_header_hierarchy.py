import os

from jet.code.markdown_utils import parse_markdown
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.utils.print_utils import print_dict_types


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_4/top_isekai_anime_2025/pages/www_ranker_com_list_best_isekai_anime_2025_anna_lindwasser/page_preprocessed.html"
    html_str: str = load_file(html_file)

    md_content = convert_html_to_markdown(html_str)

    save_file(md_content, f"{output_dir}/md_content.md")

    markdown_tokens = base_parse_markdown(md_content)
    save_file(markdown_tokens, f"{output_dir}/markdown_tokens.json")

    results = derive_by_header_hierarchy(md_content)
    save_file(results, f"{output_dir}/results.json")
