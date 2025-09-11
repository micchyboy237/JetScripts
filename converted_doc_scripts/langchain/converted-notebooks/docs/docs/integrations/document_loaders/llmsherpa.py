from jet.logger import logger
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
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
# LLM Sherpa

This notebook covers how to use `LLM Sherpa` to load files of many types. `LLM Sherpa` supports different file formats including DOCX, PPTX, HTML, TXT, and XML.

# `LLMSherpaFileLoader` use LayoutPDFReader, which is part of the LLMSherpa library. This tool is designed to parse PDFs while preserving their layout information, which is often lost when using most PDF to text parsers.

# Here are some key features of LayoutPDFReader:

* It can identify and extract sections and subsections along with their levels.
* It combines lines to form paragraphs.
* It can identify links between sections and paragraphs.
* It can extract tables along with the section the tables are found in.
* It can identify and extract lists and nested lists.
* It can join content spread across pages.
* It can remove repeating headers and footers.
* It can remove watermarks.

check [llmsherpa](https://llmsherpa.readthedocs.io/en/latest/) documentation.

`INFO: this library fail with some pdf files so use it with caution.`
"""
logger.info("# LLM Sherpa")



"""
## LLMSherpaFileLoader

Under the hood LLMSherpaFileLoader defined some strategist to load file content: ["sections", "chunks", "html", "text"], setup [nlm-ingestor](https://github.com/nlmatics/nlm-ingestor) to get `llmsherpa_api_url` or use the default.

### sections strategy: return the file parsed into sections
"""
logger.info("## LLMSherpaFileLoader")


loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="sections",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()

docs[1]

len(docs)

"""
### chunks strategy: return the file parsed into chunks
"""
logger.info("### chunks strategy: return the file parsed into chunks")


loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="chunks",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()

docs[1]

len(docs)

"""
### html strategy: return the file as one html document
"""
logger.info("### html strategy: return the file as one html document")


loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="html",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()

docs[0].page_content[:400]

len(docs)

"""
### text strategy: return the file as one text document
"""
logger.info("### text strategy: return the file as one text document")


loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="text",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()

docs[0].page_content[:400]

len(docs)

logger.info("\n\n[DONE]", bright=True)