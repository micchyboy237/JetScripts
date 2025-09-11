from jet.logger import logger
from langchain_pymupdf4llm import PyMuPDF4LLMLoader, PyMuPDF4LLMParser
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
# PyMuPDF4LLM

[PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm) is aimed to make it easier to extract PDF content in Markdown format, needed for LLM & RAG applications.

[langchain-pymupdf4llm](https://github.com/lakinduboteju/langchain-pymupdf4llm) integrates PyMuPDF4LLM to LangChain as a Document Loader.
"""
logger.info("# PyMuPDF4LLM")

# %pip install -qU langchain-pymupdf4llm


logger.info("\n\n[DONE]", bright=True)