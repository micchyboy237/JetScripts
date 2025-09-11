from jet.logger import logger
from langchain_community.document_loaders import ConcurrentLoader
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
# Concurrent Loader

Works just like the GenericLoader but concurrently for those who choose to optimize their workflow.
"""
logger.info("# Concurrent Loader")


loader = ConcurrentLoader.from_filesystem("example_data/", glob="**/*.txt")

files = loader.load()

len(files)

logger.info("\n\n[DONE]", bright=True)