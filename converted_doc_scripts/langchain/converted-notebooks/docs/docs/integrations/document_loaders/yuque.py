from jet.logger import logger
from langchain_community.document_loaders import YuqueLoader
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
# Yuque

>[Yuque](https://www.yuque.com/) is a professional cloud-based knowledge base for team collaboration in documentation.

This notebook covers how to load documents from `Yuque`.

You can obtain the personal access token by clicking on your personal avatar in the [Personal Settings](https://www.yuque.com/settings/tokens) page.
"""
logger.info("# Yuque")


loader = YuqueLoader(access_token="<your_personal_access_token>")

docs = loader.load()

logger.info("\n\n[DONE]", bright=True)