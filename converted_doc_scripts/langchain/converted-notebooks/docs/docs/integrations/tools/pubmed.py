from jet.logger import logger
from langchain_community.tools.pubmed.tool import PubmedQueryRun
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

>[PubMed®](https://pubmed.ncbi.nlm.nih.gov/) comprises more than 35 million citations for biomedical literature from `MEDLINE`, life science journals, and online books. Citations may include links to full text content from PubMed Central and publisher web sites.

This notebook goes over how to use `PubMed` as a tool.
"""
logger.info("# PubMed")

# %pip install xmltodict


tool = PubmedQueryRun()

tool.invoke("What causes lung cancer?")

logger.info("\n\n[DONE]", bright=True)