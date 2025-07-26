import shutil

from jet.code.markdown_utils import parse_markdown
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.utils.print_utils import print_dict_types

import os
OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/top_isekai_anime_2025/pages/gamerant_com_new_isekai_anime_2025/page.html"

    md_content = convert_html_to_markdown(html_file)
    save_file(md_content, f"{OUTPUT_DIR}/md_content.md")

    markdown_tokens = base_parse_markdown(html_file)
    save_file(markdown_tokens, f"{OUTPUT_DIR}/markdown_tokens.json")

    results = derive_by_header_hierarchy(html_file)
    save_file(results, f"{OUTPUT_DIR}/results.json")
