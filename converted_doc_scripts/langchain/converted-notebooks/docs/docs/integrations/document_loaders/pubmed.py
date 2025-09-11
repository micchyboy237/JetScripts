from jet.logger import logger
from langchain_community.document_loaders import PubMedLoader
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
# PubMed

>[PubMed®](https://pubmed.ncbi.nlm.nih.gov/) by `The National Center for Biotechnology Information, National Library of Medicine` comprises more than 35 million citations for biomedical literature from `MEDLINE`, life science journals, and online books. Citations may include links to full text content from `PubMed Central` and publisher web sites.
"""
logger.info("# PubMed")


loader = PubMedLoader("chatgpt")

docs = loader.load()

len(docs)

docs[1].metadata

docs[1].page_content

logger.info("\n\n[DONE]", bright=True)