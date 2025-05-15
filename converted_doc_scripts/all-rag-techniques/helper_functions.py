import json
import os
from jet.file.utils import load_file
from jet.logger import logger
import pypdf

DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/data"


def extract_text_from_pdf(pdf_path) -> str:
    pdf_path = os.path.join(DATA_DIR, "AI_Information.pdf")
    logger.debug(f"Extracting text from {pdf_path}...")
    all_text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_text += text
    return all_text


# def extract_text_from_pdf(pdf_path):
#     data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_isekai_anime.md"
#     all_text = load_file(data_path)
#     return all_text

def extract_text_from_json(json_path) -> str:
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Combine header, content, and parent_header into a single text string for each entry
    texts = []
    for item in data:
        text = f"{item.get('header', '')}\n{item.get('parent_header', '')}\n{item.get('content', '')}".strip()
        texts.append(text)
    all_text = "\n\n".join(texts)
    return all_text
