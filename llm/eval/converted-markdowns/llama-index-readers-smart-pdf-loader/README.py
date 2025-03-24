from llama_index.readers.pdf_table import PDFTableReader
from llama_index.core import VectorStoreIndex
from pathlib import Path
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()


# pip install llama-index-readers-smart-pdf-loader

# pip install llmsherpa

# Solution to fix "RuntimeError: Can not find Ghostscript library (libgs)"
# sudo cp /opt/homebrew/Cellar/ghostscript/10.04.0/lib/libgs.dylib /usr/local/lib/


# from llama_index.readers.smart_pdf_loader import SmartPDFLoader
pdf_file = "/Users/jethroestrada/Downloads/arxiv_sample.pdf"
reader = PDFTableReader()
pdf_path = Path(pdf_file)
documents = reader.load_data(file=pdf_path, pages="all")

# llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
# # also allowed is a file path e.g. /home/downloads/xyz.pdf
# pdf_url = "https://arxiv.org/pdf/1910.13461.pdf"
# pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
# documents = pdf_loader.load_data(pdf_url)


index = VectorStoreIndex.from_documents(documents, show_progress=True)
query_engine = index.as_query_engine()

response = query_engine.query("list all the tasks that work with bart")
print(response)

response = query_engine.query("what is the bart performance score on squad")
print(response)

logger.info("\n\n[DONE]", bright=True)
