import os
import shutil
from typing import List, Optional
from jet.code.markdown_utils import convert_html_to_markdown, derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.wordnet.text_chunker import chunk_texts_with_data

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def extract_doc_chunks(html: str, chunk_size: int = 200, chunk_overlap: int = 50, model: Optional[OLLAMA_MODEL_NAMES] = None) -> List[str]:
    md_content = convert_html_to_markdown(html, ignore_links=True)
    # original_docs = derive_by_header_hierarchy(md_content, ignore_links=True)
    headings = derive_by_header_hierarchy(md_content, ignore_links=True)
    docs = [f"{header["header"]}\n{header["content"]}" for header in headings if header['content']]
    chunks = chunk_texts_with_data(md_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model)
    return docs, chunks

if __name__ == "__main__":
    html = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_rag_context_engineering_tips_2025_reddit/https_www_reddit_com_r_rag_comments_1mvzwrq_context_engineering_for_advanced_rag_curious_how/page.html")

    docs, chunks = extract_doc_chunks(html)
    save_file(docs, f"{OUTPUT_DIR}/docs.json")
    save_file(chunks, f"{OUTPUT_DIR}/chunks.json")
