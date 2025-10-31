from jet.wordnet.text_chunker import truncate_texts_fast
from tqdm import tqdm
from jet.code.markdown_utils import convert_html_to_markdown, convert_markdown_to_text, derive_by_header_hierarchy
from jet.code.extraction import extract_sentences
from jet.file.utils import load_file, save_file
from jet.logger import logger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def main():
    """
    Demonstrates usage of the extract_sentences function with various text inputs.
    Optimized for Mac M1 with MPS support.
    """
    html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html")
    md_content = convert_html_to_markdown(html, ignore_links=True)
    save_file(md_content, f"{OUTPUT_DIR}/md_content.md")

    headers = derive_by_header_hierarchy(md_content, ignore_links=True)
    save_file(headers, f"{OUTPUT_DIR}/headers.json")

    header_texts = [f"{header['header']}\n\n{header['content']}" for header in headers]
    save_file(header_texts, f"{OUTPUT_DIR}/header_texts.json")

    header_texts = truncate_texts_fast(
        header_texts,
        model="qwen3-instruct-2507:4b",
        max_tokens=512,
        strict_sentences=True,
        show_progress=True
    )

    for idx, header_md_content in enumerate(tqdm(header_texts, desc="Extracting RAG sentences", unit="header")):
        header_dir = os.path.join(OUTPUT_DIR, f"header_{idx + 1}")
        os.makedirs(header_dir, exist_ok=True)

        save_file(header_md_content, f"{header_dir}/rag_markdown.md")

        text = convert_markdown_to_text(header_md_content)
        save_file(text, f"{header_dir}/rag_text.txt")

        # Optional: nested progress tracking if extract_sentences is slow
        sentences = extract_sentences(text, use_gpu=True, valid_only=True)

        if sentences:
            save_file(sentences, f"{header_dir}/rag_sentences.json")
        else:
            logger.warning(f"No valid sentences for header {idx + 1}")

if __name__ == "__main__":
    main()
