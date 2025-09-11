from jet.logger import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.merge import MergedDataLoader
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
# Merge Documents Loader

Merge the documents returned from a set of specified data loaders.
"""
logger.info("# Merge Documents Loader")


loader_web = WebBaseLoader(
    "https://github.com/basecamp/handbook/blob/master/37signals-is-you.md"
)


loader_pdf = PyPDFLoader("../MachineLearning-Lecture01.pdf")


loader_all = MergedDataLoader(loaders=[loader_web, loader_pdf])

docs_all = loader_all.load()

len(docs_all)

logger.info("\n\n[DONE]", bright=True)