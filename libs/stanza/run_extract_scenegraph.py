from jet.wordnet.text_chunker import truncate_texts
from stanza.server import CoreNLPClient
from tqdm import tqdm
from jet.code.markdown_utils import convert_html_to_markdown, convert_markdown_to_text, derive_by_header_hierarchy
from jet.code.extraction import extract_sentences
from jet.file.utils import load_file, save_file
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

    with CoreNLPClient(preload=False) as client:
        for idx, header_md_content in enumerate(tqdm(header_texts, desc="Extracting RAG sentences", unit="header")):
            header_dir = os.path.join(OUTPUT_DIR, f"header_{idx + 1}")
            os.makedirs(header_dir, exist_ok=True)

            save_file(header_md_content, f"{header_dir}/rag_markdown.md")

            text = convert_markdown_to_text(header_md_content)
            save_file(text, f"{header_dir}/rag_text.txt")

            # Optional: nested progress tracking if extract_sentences is slow
            sentences = extract_sentences(text, use_gpu=True)
            save_file(sentences, f"{header_dir}/rag_sentences.json")

            truncated_sents = truncate_texts(sentences, max_tokens=128)
            save_file(truncated_sents, f"{header_dir}/truncated_sentences.json")

            for sentence_idx, sentence in enumerate(tqdm(truncated_sents, desc="Processing documents", unit="doc")):
                scenegraph = client.scenegraph(sentence)

                output_path = f"{header_dir}/scenegraph/scenegraph_{sentence_idx + 1}.json"
                save_file(scenegraph, output_path)

if __name__ == "__main__":
    main()
