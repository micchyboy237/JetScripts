from jet.code.markdown_utils import convert_html_to_markdown
from jet.vectors.extraction import extract_paragraphs

from jet.transformers.formatters import format_json
from jet.file.utils import load_file, save_file
from jet.logger import logger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def main():
    """
    Demonstrates usage of the extract_paragraphs function with various text inputs.
    Optimized for Mac M1 with MPS support.
    """
    html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html")
    md_content = convert_html_to_markdown(html, ignore_links=True)
    save_file(md_content, f"{OUTPUT_DIR}/md_content.md")

    text = "First paragraph without breaks. It continues here. Second paragraph starts semantically. More content follows."
    paragraphs = extract_paragraphs(text, use_gpu=True)
    
    logger.success(format_json(paragraphs))
    save_file(paragraphs, f"{OUTPUT_DIR}/paragraphs.json")

if __name__ == "__main__":
    main()