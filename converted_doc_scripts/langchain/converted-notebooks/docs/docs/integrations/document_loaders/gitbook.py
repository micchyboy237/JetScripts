from jet.logger import logger
from langchain_community.document_loaders import GitbookLoader
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
# GitBook

>[GitBook](https://docs.gitbook.com/) is a modern documentation platform where teams can document everything from products to internal knowledge bases and APIs.

This notebook shows how to pull page data from any `GitBook`.
"""
logger.info("# GitBook")


"""
### Load from single GitBook page
"""
logger.info("### Load from single GitBook page")

loader = GitbookLoader("https://docs.gitbook.com")

page_data = loader.load()

page_data

"""
### Load from all paths in a given GitBook
For this to work, the GitbookLoader needs to be initialized with the root path (`https://docs.gitbook.com` in this example) and have `load_all_paths` set to `True`.
"""
logger.info("### Load from all paths in a given GitBook")

loader = GitbookLoader("https://docs.gitbook.com", load_all_paths=True)
all_pages_data = loader.load()

logger.debug(f"fetched {len(all_pages_data)} documents.")
all_pages_data[2]

logger.info("\n\n[DONE]", bright=True)