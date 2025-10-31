from stanza.server import CoreNLPClient
from tqdm import tqdm
# from jet.libs.bertopic.examples.mock import load_sample_data
from jet.code.markdown_utils import convert_markdown_to_text
from jet.code.extraction import extract_sentences
from jet.file.utils import load_file, save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/run_format_html/headings.json"
headings = load_file(file_path)
docs = [h["text"] for h in headings]
# docs = load_sample_data(model="embeddinggemma", chunk_size=200, truncate=True)

def main():
    """
    Demonstrates usage of the extract_sentences function with various text inputs.
    Optimized for Mac M1 with MPS support.
    """
    save_file(docs, f"{OUTPUT_DIR}/docs.json")

    with CoreNLPClient(preload=False) as client:
        for idx, md_content in enumerate(tqdm(docs, desc="Extracting RAG sentences", unit="header")):
            header_dir = os.path.join(OUTPUT_DIR, f"header_{idx + 1}")
            os.makedirs(header_dir, exist_ok=True)

            save_file(md_content, f"{header_dir}/rag_markdown.md")

            text = convert_markdown_to_text(md_content)
            save_file(text, f"{header_dir}/rag_text.txt")

            # Optional: nested progress tracking if extract_sentences is slow
            sentences = extract_sentences(text, use_gpu=True)
            save_file(sentences, f"{header_dir}/rag_sentences.json")

            for sentence_idx, sentence in enumerate(tqdm(sentences, desc="Processing documents", unit="doc")):
                scenegraph = client.scenegraph(sentence)

                output_path = f"{header_dir}/scenegraph/scenegraph_{sentence_idx + 1}.json"
                save_file(scenegraph, output_path)

if __name__ == "__main__":
    main()
