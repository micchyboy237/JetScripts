import os
import shutil
import spacy
from typing import List
from jet.code.html_utils import convert_dl_blocks_to_md
from jet.file.utils import load_file, save_file
from jet.logger import logger
from span_marker import SpanMarkerModel
from jet.scrapers.header_hierarchy import HtmlHeaderDoc, extract_header_hierarchy
from jet.wordnet.text_chunker import chunk_texts
from tqdm import tqdm

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

model = SpanMarkerModel.from_pretrained(
    "tomaarsen/span-marker-roberta-large-ontonotes5",
).to("mps")

# Load the spaCy model
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("span_marker", config={
    "model": "tomaarsen/span-marker-roberta-large-ontonotes5",
    "device": "mps",
})

def extract_nlp(text: str) -> dict:
    logger.info("Running span marker NER extraction")
    span_marker_entities = model.predict(text)

    logger.info("Running spacy NER extraction")
    doc = nlp(text)
    spacy_entities = doc.ents

    return {
        # "pos": pos,
        # "sentences": sentences,
        "span_marker_entities": span_marker_entities,
        "spacy_entities": spacy_entities,
        # "dependencies": dependencies,
        # "constituencies": constituencies,
        # "scenes": scenes,
        # "sentence_details": sentence_details,
    }

if __name__ == "__main__":
    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"

    html_str: str = load_file(html_file)
    html_str = convert_dl_blocks_to_md(html_str)
    save_file(html_str, f"{OUTPUT_DIR}/page.html")

    headings: List[HtmlHeaderDoc] = extract_header_hierarchy(html_str)
    save_file(headings, f"{OUTPUT_DIR}/headings.json")

    for idx, heading in enumerate(tqdm(headings, desc="Processing headings...")):
        header = heading["header"]
        content = heading["content"]

        if not content:
            continue

        sub_output_dir = f"{OUTPUT_DIR}/heading_{idx + 1}"
        chunks = chunk_texts(
            content,
            chunk_size=512,
            chunk_overlap=50,
            model="qwen3-instruct-2507:4b",
        )
        for chunk_idx, chunk in enumerate(chunks):
            results = extract_nlp(chunk)
            for key, nlp_results in results.items():
                save_file(nlp_results, f"{sub_output_dir}/chunk_{chunk_idx + 1}/{key}_results.json")
