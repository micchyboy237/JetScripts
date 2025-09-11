from jet.logger import logger
from langchain_community.document_loaders import CollegeConfidentialLoader
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
# College Confidential

>[College Confidential](https://www.collegeconfidential.com/) gives information on 3,800+ colleges and universities.

This covers how to load `College Confidential` webpages into a document format that we can use downstream.
"""
logger.info("# College Confidential")


loader = CollegeConfidentialLoader(
    "https://www.collegeconfidential.com/colleges/brown-university/"
)

data = loader.load()

data

logger.info("\n\n[DONE]", bright=True)