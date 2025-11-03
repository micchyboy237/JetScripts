import os
import shutil
from typing import List
from jet.code.extraction import extract_sentences
from jet.code.html_utils import convert_dl_blocks_to_md
from jet.file.utils import load_file, save_file
from jet.scrapers.header_hierarchy import HtmlHeaderDoc, extract_header_hierarchy
from jet.wordnet.text_chunker import chunk_texts
from tqdm import tqdm

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"

    html_str: str = load_file(html_file)
    html_str = convert_dl_blocks_to_md(html_str)
    save_file(html_str, f"{OUTPUT_DIR}/page.html")

    headings: List[HtmlHeaderDoc] = extract_header_hierarchy(html_str)
    save_file(headings, f"{OUTPUT_DIR}/headings.json")

    sentences = []
    for heading in tqdm(headings, desc="Processing headings..."):
        content = heading["content"]
        chunks = chunk_texts(
            content,
            chunk_size=512,
            chunk_overlap=50,
            model="qwen3-instruct-2507:4b",
        )
        for chunk in chunks:
            chunk_sentences = extract_sentences(chunk, use_gpu=True, valid_only=True)
            sentences.extend(chunk_sentences)
            save_file(sentences, f"{OUTPUT_DIR}/sentences.json")
    
    save_file(sentences, f"{OUTPUT_DIR}/sentences.json")
