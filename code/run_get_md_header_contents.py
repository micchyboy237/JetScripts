import os
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.file.utils import load_file, save_file
from jet.logger import logger


from typing import List, Dict

from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/gamerant.com/new_isekai_anime_2025/docs.json"
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/gamerant.com/new_isekai_anime_2025/page.html"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    html = load_file(html_file)
    results = get_md_header_contents(
        html, ignore_links=True, base_url=docs["source_url"])
    results = [{
        "parent_header": result["parent_header"],
        "header": result["header"],
        "content": result["content"],
        "text": result["text"],
        "links": result["links"],
    } for result in results]
    save_file(results, f"{output_dir}/results.json")
