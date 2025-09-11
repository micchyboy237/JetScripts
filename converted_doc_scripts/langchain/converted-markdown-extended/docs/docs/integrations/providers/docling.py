from jet.logger import logger
from langchain_docling import DoclingLoader
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
# Docling

> [Docling](https://github.com/DS4SD/docling) parses PDF, DOCX, PPTX, HTML, and other formats into a rich unified representation including document layout, tables etc., making them ready for generative AI workflows like RAG.
>
> This integration provides Docling's capabilities via the `DoclingLoader` document loader.

## Installation and Setup

Simply install `langchain-docling` from your package manager, e.g. pip:
"""
logger.info("# Docling")

pip install langchain-docling

"""
## Document Loader

The `DoclingLoader` class in `langchain-docling` seamlessly integrates Docling into
LangChain, enabling you to:
- use various document types in your LLM applications with ease and speed, and
- leverage Docling's rich representation for advanced, document-native grounding.

Basic usage looks as follows:
"""
logger.info("## Document Loader")


FILE_PATH = ["https://arxiv.org/pdf/2408.09869"]  # Docling Technical Report

loader = DoclingLoader(file_path=FILE_PATH)

docs = loader.load()

"""
For end-to-end usage check out
[this example](/docs/integrations/document_loaders/docling).

## Additional Resources

- [LangChain Docling integration GitHub](https://github.com/DS4SD/docling-langchain)
- [LangChain Docling integration PyPI package](https://pypi.org/project/langchain-docling/)
- [Docling GitHub](https://github.com/DS4SD/docling)
- [Docling docs](https://ds4sd.github.io/docling/)
"""
logger.info("## Additional Resources")

logger.info("\n\n[DONE]", bright=True)