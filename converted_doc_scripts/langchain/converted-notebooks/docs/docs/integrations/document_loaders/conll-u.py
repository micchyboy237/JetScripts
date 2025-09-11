from jet.logger import logger
from langchain_community.document_loaders import CoNLLULoader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# CoNLL-U

>[CoNLL-U](https://universaldependencies.org/format.html) is revised version of the CoNLL-X format. Annotations are encoded in plain text files (UTF-8, normalized to NFC, using only the LF character as line break, including an LF character at the end of file) with three types of lines:
>- Word lines containing the annotation of a word/token in 10 fields separated by single tab characters; see below.
>- Blank lines marking sentence boundaries.
>- Comment lines starting with hash (#).

This is an example of how to load a file in [CoNLL-U](https://universaldependencies.org/format.html) format. The whole file is treated as one document. The example data (`conllu.conllu`) is based on one of the standard UD/CoNLL-U examples.
"""
logger.info("# CoNLL-U")


loader = CoNLLULoader("example_data/conllu.conllu")

document = loader.load()

document

logger.info("\n\n[DONE]", bright=True)