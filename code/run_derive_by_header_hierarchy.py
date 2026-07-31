import os
import shutil

from jet.code.html_utils import convert_dl_blocks_to_md
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import load_file, save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/smolagents/tools/generated/visit_webpage_tool/visit_webpage_tool_logs/call_0001/page.html"
    html = load_file(html_file)
    html = convert_dl_blocks_to_md(html)

    results_ignore_links = derive_by_header_hierarchy(
        html, ignore_links=True, valid_sentences_only=False
    )
    save_file(results_ignore_links, f"{OUTPUT_DIR}/results_ignore_links.json")

    results_with_links = derive_by_header_hierarchy(
        html, ignore_links=False, valid_sentences_only=False
    )
    save_file(results_with_links, f"{OUTPUT_DIR}/results_with_links.json")

    results_ignore_links = derive_by_header_hierarchy(
        html, ignore_links=True, valid_sentences_only=True
    )
    save_file(
        results_ignore_links, f"{OUTPUT_DIR}/results_ignore_links_valid_sents.json"
    )

    results_with_links = derive_by_header_hierarchy(
        html, ignore_links=False, valid_sentences_only=True
    )
    save_file(results_with_links, f"{OUTPUT_DIR}/results_with_links_valid_sents.json")
