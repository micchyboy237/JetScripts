from jet.logger import logger
from langchain_text_splitters import CharacterTextSplitter
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
---
keywords: [charactertextsplitter]
---

# How to split by character

This is the simplest method. This [splits](/docs/concepts/text_splitters/) based on a given character sequence, which defaults to `"\n\n"`. Chunk length is measured by number of characters.

1. How the text is split: by single character separator.
2. How the chunk size is measured: by number of characters.

To obtain the string content directly, use `.split_text`.

To create LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects (e.g., for use in downstream tasks), use `.create_documents`.
"""
logger.info("# How to split by character")

# %pip install -qU langchain-text-splitters


with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([state_of_the_union])
logger.debug(texts[0])

"""
Use `.create_documents` to propagate metadata associated with each document to the output chunks:
"""
logger.info("Use `.create_documents` to propagate metadata associated with each document to the output chunks:")

metadatas = [{"document": 1}, {"document": 2}]
documents = text_splitter.create_documents(
    [state_of_the_union, state_of_the_union], metadatas=metadatas
)
logger.debug(documents[0])

"""
Use `.split_text` to obtain the string content directly:
"""
logger.info("Use `.split_text` to obtain the string content directly:")

text_splitter.split_text(state_of_the_union)[0]

logger.info("\n\n[DONE]", bright=True)