import os
import shutil

from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.file.utils import load_file, save_file

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html")
    
    md_content_ignore_links = convert_html_to_markdown(html, ignore_links=True)
    save_file(md_content_ignore_links, f"{OUTPUT_DIR}/md_content_ignore_links.md")
    
    md_content_with_links = convert_html_to_markdown(html, ignore_links=False)
    save_file(md_content_with_links, f"{OUTPUT_DIR}/md_content_with_links.md")

    # md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/code/extraction/generated/run_extract_notebook_texts/GenAI_Agents/docs/Academic_Task_Learning_Agent_LangGraph.md"
    # md_content: str = load_file(md_file)

    results_ignore_links = derive_by_header_hierarchy(md_content_ignore_links, valid_sentences_only=True)
    save_file(results_ignore_links, f"{OUTPUT_DIR}/results_ignore_links.json")

    results_with_links = derive_by_header_hierarchy(md_content_with_links, valid_sentences_only=True)
    save_file(results_with_links, f"{OUTPUT_DIR}/results_with_links.json")
