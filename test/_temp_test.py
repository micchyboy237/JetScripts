from pathlib import Path

import trafilatura
from jet.file.utils import save_file
from jet.wordnet.text_chunker import chunk_texts

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

# Load HTML content from the specified file
html_file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/vectors/semantic_search/generated/run_web_search/top_isekai_anime_2026/pages/gamerant_com_new_isekai_anime_2026/page.html"
with open(html_file_path, encoding="utf-8") as file:
    html = file.read()

text = trafilatura.extract(html)
print(f"\nHTML Text:\n{text}")

chunks = chunk_texts(text, strict_sentences=True)
print(f"Chunks: {len(chunks)}")

save_file(text, OUTPUT_DIR / "text.md")
save_file(chunks, OUTPUT_DIR / "chunks.json")
