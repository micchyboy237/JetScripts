from jet.logger import logger
from langchain_community.document_loaders import AsyncHtmlLoader
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
# AsyncHtml

`AsyncHtmlLoader` loads raw HTML from a list of URLs concurrently.
"""
logger.info("# AsyncHtml")


urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

docs[0].page_content[1000:2000]

docs[1].page_content[1000:2000]

logger.info("\n\n[DONE]", bright=True)