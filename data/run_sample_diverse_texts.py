
import os
import shutil
from typing import List
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.data.sample_diverse_texts import sample_diverse_texts
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import count_tokens
from jet.transformers.formatters import format_json
from jet.wordnet.text_chunker import chunk_texts_with_data

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_4/top_isekai_anime_2025/docs.json"

    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    chunk_size = 500
    chunk_overlap = 100

    header_docs: List[HeaderDoc] = load_file(docs_file)["documents"]

    chunk_texts = []
    for header_doc in header_docs:
        doc_index = header_doc["doc_index"]
        header = header_doc["header"]
        content = header_doc['content']
        source = header_doc.get("source", "Unknown Source")
        parent_header = header_doc.get("parent_header", None)
        buffer: int = count_tokens(embed_model, header)
        doc_ids = [header_doc["id"]]
        chunks = chunk_texts_with_data(content, chunk_size,
                                       chunk_overlap, embed_model, doc_ids, buffer)
        chunk_texts.extend([
            f"{header}\n{chunk["content"]}"
            for chunk in chunks
        ])

    save_file(chunk_texts, f"{OUTPUT_DIR}/chunk_texts.json")

    results: List[str] = sample_diverse_texts(chunk_texts)

    logger.gray(f"Results: ({len(results)})")
    logger.success(format_json(results))

    save_file(results, f"{OUTPUT_DIR}/results.json")
