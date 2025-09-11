from jet.logger import logger
from langchain_community.document_loaders import MHTMLLoader
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
# mhtml

MHTML is a is used both for emails but also for archived webpages. MHTML, sometimes referred as MHT, stands for MIME HTML is a single file in which entire webpage is archived. When one saves a webpage as MHTML format, this file extension will contain HTML code, images, audio files, flash animation etc.
"""
logger.info("# mhtml")


loader = MHTMLLoader(
    file_path="../../../../../../tests/integration_tests/examples/example.mht"
)

documents = loader.load()

for doc in documents:
    logger.debug(doc)

logger.info("\n\n[DONE]", bright=True)