from jet.code.markdown_utils._converters import convert_markdown_to_text
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data
from jet.logger import logger
from typing import List
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def preprocess_for_rag(md_docs: List[str]) -> List[str]:
    return [convert_markdown_to_text(doc) for doc in md_docs]

# Test with pytest
if __name__ == "__main__":
    md_docs = load_sample_data(model="embeddinggemma", chunk_size=512, chunk_overlap=64)
    results = preprocess_for_rag(md_docs)

    logger.success(results)

    save_file(results, f"{OUTPUT_DIR}/results.json")